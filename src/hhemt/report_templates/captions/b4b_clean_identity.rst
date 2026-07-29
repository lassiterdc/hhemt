Raw byte-for-byte identity across clean compute configs.

Per-timestep binary identity of the raw TRITON output rasters (``out_triton/bin`` water-level / max-water-level grids) across this master's clean (``n_resumes == 0``) compute-config sub-analyses, compared against a single clean reference config. Green cells are byte-identical to the reference at that reporting timestep; crimson cells differ. This is a clean-vs-clean comparison (no resume boundaries). A ``passed = False`` verdict with differing cells is EXPECTED where the clean configs span backends (GPU vs CPU raw rasters are not bit-identical — BIT4BIT is a double-precision serial-oracle property) or rank counts; the verdict summary discloses the differing-cell denominator and cause. This is cross-config divergence, NOT resume-induced — the resume-validity byte-identity claim is the companion ``b4b_clean_vs_resume`` figure. Reads ``eda/b4b_clean_identity.zarr``; renders an honest-degradation panel when that backing artifact is absent or degraded.

**Sources:**

{{ snakemake.params.source_paths_rst }}
