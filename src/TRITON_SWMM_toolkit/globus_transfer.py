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
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import globus_sdk

from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec

# Client ID for this application registered at developers.globus.org.
# This is a public identifier — it does not grant any special privileges.
# TODO: Replace with the real client ID after registering a native app at
# https://developers.globus.org — see /setup-hpc-integration Step 3.
_GLOBUS_CLIENT_ID = "REPLACE_WITH_REGISTERED_CLIENT_ID"

# Globus transfer task polling interval (seconds)
_POLL_INTERVAL_S = 10

# Sim output subdirectories whose bulk data we skip by default when
# transferring results locally.  Override by passing filter_sim_outputs=False.
_SIM_OUTPUT_EXCLUDE_PATTERNS = [
    "*.inp",  # large intermediate SWMM input files
]


class GlobusTransferManager:
    """Authenticates with Globus and submits transfer tasks.

    Attributes:
        transfer_client: Authenticated :class:`globus_sdk.TransferClient`.
    """

    def __init__(self) -> None:
        self.transfer_client = self._get_authenticated_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        spec: GlobusTransferSpec,
        filter_sim_outputs: bool = False,
    ) -> str:
        """Submit all items in *spec* as a single Globus transfer task.

        Args:
            spec:               Transfer specification loaded from YAML.
            filter_sim_outputs: If True, exclude large intermediate sim files
                                (see ``_SIM_OUTPUT_EXCLUDE_PATTERNS``).

        Returns:
            Globus task ID string.  Pass to :meth:`wait` to block until done.
        """
        tdata = globus_sdk.TransferData(
            self.transfer_client,
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

        response = self.transfer_client.submit_transfer(tdata)
        task_id: str = response["task_id"]
        print(f"[Globus] Transfer submitted — task_id={task_id}", flush=True)
        return task_id

    def wait(
        self,
        task_id: str,
        polling_interval: int = _POLL_INTERVAL_S,
        timeout_minutes: Optional[int] = None,
    ) -> None:
        """Block until Globus task *task_id* completes or raises on failure.

        Args:
            task_id:          Task ID returned by :meth:`transfer`.
            polling_interval: Seconds between status checks.
            timeout_minutes:  Raise :exc:`TimeoutError` if task exceeds this.

        Raises:
            RuntimeError:  If the transfer task fails or is cancelled.
            TimeoutError:  If *timeout_minutes* is set and elapsed.
        """
        deadline = (
            time.time() + timeout_minutes * 60 if timeout_minutes else None
        )
        while True:
            task = self.transfer_client.get_task(task_id)
            status = task["status"]
            print(f"[Globus] Task {task_id} status: {status}", flush=True)

            if status == "SUCCEEDED":
                print(f"[Globus] Transfer complete.", flush=True)
                return
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"Globus transfer {task_id} ended with status={status}. "
                    f"View details at https://app.globus.org/activity/{task_id}"
                )

            if deadline and time.time() > deadline:
                raise TimeoutError(
                    f"Globus transfer {task_id} did not complete within "
                    f"{timeout_minutes} minutes."
                )

            time.sleep(polling_interval)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _get_authenticated_client() -> globus_sdk.TransferClient:
        """Return an authenticated TransferClient using native app flow.

        Credentials are cached locally by globus-sdk's token storage.
        On first run this opens a browser for OAuth2 login.
        """
        client = globus_sdk.NativeAppAuthClient(_GLOBUS_CLIENT_ID)
        token_storage = globus_sdk.tokenstorage.SimpleJSONFileAdapter(
            Path.home() / ".globus_tokens.json"
        )

        # Try to load cached tokens first
        try:
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                token_storage.get_token_data(
                    globus_sdk.TransferClient.scopes.all
                )["refresh_token"],
                client,
                access_token=token_storage.get_token_data(
                    globus_sdk.TransferClient.scopes.all
                )["access_token"],
                expires_at=token_storage.get_token_data(
                    globus_sdk.TransferClient.scopes.all
                )["expires_at_seconds"],
                on_refresh=lambda r: token_storage.store(r),
            )
        except Exception:
            # No cached tokens — run interactive login
            client.oauth2_start_flow(refresh_tokens=True)
            authorize_url = client.oauth2_get_authorize_url()
            print(f"[Globus] Please log in:\n{authorize_url}", flush=True)
            auth_code = input("Enter the auth code: ").strip()
            token_response = client.oauth2_exchange_code_for_tokens(auth_code)
            token_storage.store(token_response)
            transfer_tokens = token_response.by_resource_server[
                "transfer.api.globus.org"
            ]
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                transfer_tokens["refresh_token"],
                client,
                access_token=transfer_tokens["access_token"],
                expires_at=transfer_tokens["expires_at_seconds"],
                on_refresh=lambda r: token_storage.store(r),
            )

        return globus_sdk.TransferClient(authorizer=authorizer)

    @staticmethod
    def _deadline_str(minutes: Optional[int]) -> Optional[str]:
        """Convert deadline minutes to ISO 8601 string for Globus API."""
        if minutes is None:
            return None
        import datetime

        deadline = datetime.datetime.utcnow() + datetime.timedelta(
            minutes=minutes
        )
        return deadline.strftime("%Y-%m-%dT%H:%M:%S")
