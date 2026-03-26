"""Globus transfer manager for moving HPC simulation results.

Usage::

    from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager
    from TRITON_SWMM_toolkit.config.loaders import load_transfer_config

    spec = load_transfer_config(Path("configs/transfers/my_transfer.yaml"))
    manager = GlobusTransferManager()
    task_id = manager.transfer(spec)
    manager.wait(task_id)

Authentication:
    On first use, ``GlobusTransferManager()`` will open a browser for Globus
    OAuth2 login.  Credentials are cached locally by globus-sdk so subsequent
    runs are non-interactive.  Run the ``/setup-hpc-integration`` skill for
    a guided setup walkthrough.

    For endpoints with domain-restricted access policies (e.g. OLCF DTN,
    which requires ``sso.ccs.ornl.gov`` identities), pass the required domain
    to the constructor::

        manager = GlobusTransferManager(
            collection_uuids=["36d521b3-..."],
            session_required_domains=["sso.ccs.ornl.gov"],
        )

    This causes the Globus Auth flow to force authentication through the
    correct identity provider.  Without this, Globus Auth auto-selects the
    active browser session identity (often the wrong one) and the Transfer
    API returns a 403 ``PermissionDenied`` on ``submit_transfer``.
"""

from __future__ import annotations

import time
from pathlib import Path

import globus_sdk

from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec

# Client ID registered at developers.globus.org for the TRITON-SWMM Toolkit
# native app. This is a public identifier — it does not grant any privileges.
_GLOBUS_CLIENT_ID = "f52d5f54-fc84-4086-90ff-69ddc30c9334"

# Globus transfer task polling interval (seconds)
_POLL_INTERVAL_S = 10

# Sim output subdirectories whose bulk data we skip by default when
# transferring results locally.  Override by passing filter_sim_outputs=False.
_SIM_OUTPUT_EXCLUDE_PATTERNS = [
    "*.inp",  # large intermediate SWMM input files
]


class GlobusTransferManager:
    """Authenticates with Globus and submits transfer tasks.

    Args:
        collection_uuids: UUIDs of Globus 5 mapped collections that require
            ``data_access`` consent (e.g. UVA Standard Security Storage,
            Frontier OLCF DTN).  Globus Connect Personal endpoints do NOT
            need to be listed here — only HPC-side mapped collections do.
        session_required_domains: Domain(s) that Globus Auth must use for
            authentication, e.g. ``["sso.ccs.ornl.gov"]`` for OLCF endpoints.
            When provided, the OAuth2 authorize URL is constructed with
            ``session_required_single_domain`` and ``prompt=login``, forcing
            the user through the correct identity provider.  If omitted, Globus
            Auth uses whatever browser session is active, which may not satisfy
            the endpoint's resource policy.

    Attributes:
        transfer_client: Authenticated :class:`globus_sdk.TransferClient`.
    """

    def __init__(
        self,
        collection_uuids: list[str] | None = None,
        session_required_domains: list[str] | None = None,
    ) -> None:
        self._collection_uuids = collection_uuids or []
        self.transfer_client = self._get_authenticated_client(
            self._collection_uuids,
            session_required_domains=session_required_domains,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        spec: GlobusTransferSpec,
        filter_sim_outputs: bool = False,
        exclude_dirs: list[str] | None = None,
    ) -> str:
        """Submit all items in *spec* as a single Globus transfer task.

        Args:
            spec:               Transfer specification.
            filter_sim_outputs: If True, exclude large intermediate sim files
                                (see ``_SIM_OUTPUT_EXCLUDE_PATTERNS``).
            exclude_dirs:       Directory path suffixes to exclude via Globus
                                filter rules (e.g. ``["subanalyses/"]``).

        Returns:
            Globus task ID string.  Pass to :meth:`wait` to block until done.
        """
        tdata = globus_sdk.TransferData(
            spec.endpoints.source_uuid,
            spec.endpoints.destination_uuid,
            label=spec.label,
            sync_level=spec.sync_level,
            notify_on_succeeded=spec.notify_on_succeeded,
            notify_on_failed=spec.notify_on_failed,
            deadline=self._deadline_str(spec.deadline_minutes),
        )

        for item in spec.items:
            tdata.add_item(
                item.source_path,
                item.destination_path,
                recursive=item.recursive,
            )

        if filter_sim_outputs:
            for pattern in _SIM_OUTPUT_EXCLUDE_PATTERNS:
                tdata.add_filter_rule(pattern, method="exclude", type="file")

        if exclude_dirs:
            for pattern in exclude_dirs:
                tdata.add_filter_rule(pattern, method="exclude", type="dir")

        try:
            response = self.transfer_client.submit_transfer(tdata)
        except globus_sdk.TransferAPIError as err:
            authz = err.info.authorization_parameters
            if not authz:
                raise
            # Token was issued for a wrong/insufficient identity.  Re-authenticate
            # with the domain(s) demanded by the endpoint policy, then retry once.
            domains = authz.session_required_single_domain or []
            print(
                f"[Globus] Session policy requires domain(s): {domains}. " "Re-authenticating...",
                flush=True,
            )
            self.transfer_client = self._get_authenticated_client(
                self._collection_uuids,
                session_required_domains=domains,
                force_login=True,
            )
            response = self.transfer_client.submit_transfer(tdata)
        task_id: str = response["task_id"]
        print(f"[Globus] Transfer submitted — task_id={task_id}", flush=True)
        return task_id

    def wait(
        self,
        task_id: str,
        polling_interval: int = _POLL_INTERVAL_S,
        timeout_minutes: int | None = None,
    ) -> None:
        """Block until Globus task *task_id* completes or raises on failure.

        Args:
            task_id:          Task ID returned by :meth:`transfer`.
            polling_interval: Seconds between status checks.
            timeout_minutes:  Raise :exc:`TimeoutError` if task exceeds this.

        Raises:
            GlobusTransferError: If the transfer task fails or is cancelled.
            TimeoutError:        If *timeout_minutes* is set and elapsed.
        """
        deadline = time.time() + timeout_minutes * 60 if timeout_minutes else None
        while True:
            task = self.transfer_client.get_task(task_id)
            status = task["status"]
            print(f"[Globus] Task {task_id} status: {status}", flush=True)

            if status == "SUCCEEDED":
                print("[Globus] Transfer complete.", flush=True)
                return
            if status in ("FAILED", "CANCELLED"):
                from TRITON_SWMM_toolkit.exceptions import GlobusTransferError

                raise GlobusTransferError(task_id=task_id, status=status)

            if deadline and time.time() > deadline:
                raise TimeoutError(f"Globus transfer {task_id} did not complete within " f"{timeout_minutes} minutes.")

            time.sleep(polling_interval)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _get_authenticated_client(
        collection_uuids: list[str],
        session_required_domains: list[str] | None = None,
        force_login: bool = False,
    ) -> globus_sdk.TransferClient:
        """Return an authenticated TransferClient using native app flow.

        Credentials are cached locally by globus-sdk's token storage.
        On first run this opens a browser for OAuth2 login.

        Args:
            collection_uuids: UUIDs of Globus 5 mapped collections that require
                ``data_access`` consent.  Added as dependent scopes on the
                transfer scope so the API accepts requests against those
                collections.
            session_required_domains: If provided, the Globus Auth authorize URL
                will include ``session_required_single_domain`` for each domain,
                forcing the user to authenticate with an identity from one of
                those domains.  Required for domain-restricted endpoints such as
                the OLCF DTN (``sso.ccs.ornl.gov``).
            force_login: If True, delete any cached tokens and run a fresh
                interactive login regardless of cached state.  Use this when a
                previously cached token was issued for the wrong identity.
        """
        from globus_sdk.scopes import GCSCollectionScopes, Scope
        from globus_sdk.token_storage import JSONTokenStorage

        _TRANSFER_RS = "transfer.api.globus.org"

        # Build transfer scope, adding data_access for each mapped collection
        transfer_scope = Scope(str(globus_sdk.TransferClient.scopes.all))
        for uuid in collection_uuids:
            transfer_scope = transfer_scope.with_dependency(GCSCollectionScopes(uuid).data_access)

        client = globus_sdk.NativeAppAuthClient(_GLOBUS_CLIENT_ID)
        token_file = Path.home() / ".globus_tokens.json"
        token_storage = JSONTokenStorage(token_file)

        if force_login:
            # Wipe the cached file so stale/wrong-identity tokens are not reused.
            token_file.unlink(missing_ok=True)

        # Try to load cached tokens first (skipped when force_login wiped the file)
        cached = token_storage.get_token_data(_TRANSFER_RS)
        if cached is not None and cached.refresh_token is not None:
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                cached.refresh_token,
                client,
                access_token=cached.access_token,
                expires_at=cached.expires_at_seconds,
                on_refresh=lambda r: token_storage.store_token_response(r),
            )
        else:
            # No cached tokens — run interactive login.
            # Pass session_required_single_domain so Globus Auth forces the user
            # through the correct identity provider (e.g. sso.ccs.ornl.gov for
            # OLCF endpoints) rather than auto-selecting the active browser session.
            client.oauth2_start_flow(transfer_scope, refresh_tokens=True)
            need_forced_login = bool(session_required_domains or force_login)
            authorize_url = client.oauth2_get_authorize_url(
                session_required_single_domain=session_required_domains or [],
                **({"prompt": "login"} if need_forced_login else {}),
            )
            print(f"[Globus] Please log in:\n{authorize_url}", flush=True)
            auth_code = input("Enter the auth code: ").strip()
            token_response = client.oauth2_exchange_code_for_tokens(auth_code)
            token_storage.store_token_response(token_response)
            transfer_tokens = token_response.by_resource_server[_TRANSFER_RS]
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                transfer_tokens["refresh_token"],
                client,
                access_token=transfer_tokens["access_token"],
                expires_at=transfer_tokens["expires_at_seconds"],
                on_refresh=lambda r: token_storage.store_token_response(r),
            )

        return globus_sdk.TransferClient(authorizer=authorizer)

    @staticmethod
    def _deadline_str(minutes: int | None) -> str | None:
        """Convert deadline minutes to ISO 8601 string for Globus API."""
        if minutes is None:
            return None
        import datetime

        deadline = datetime.datetime.utcnow() + datetime.timedelta(minutes=minutes)
        return deadline.strftime("%Y-%m-%dT%H:%M:%S")
