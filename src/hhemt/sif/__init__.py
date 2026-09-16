"""hhemt.sif: identity-addressed SIF creation and resolution (SIF quest, ADR-21).

One identity type (``identity.SifIdentity``), one resolver (``identity.resolve_sif``), one
place ``apptainer build`` runs (``transaction.build_transaction``). Nothing here reads an
analysis directory; the build DAG lives under ``ContainerSpec.sif_root``.
"""
