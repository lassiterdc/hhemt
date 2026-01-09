from TRITON_SWMM_toolkit.experiment import TRITONSWMM_experiment
from TRITON_SWMM_toolkit.processing_simulation import (
    convert_coords_to_dtype,
    convert_datavars_to_dtype,
)
import sys
import pandas as pd
import xarray as xr
import numpy as np


class TRITONSWMM_sim_post_processing:
    def __init__(self, exp: TRITONSWMM_experiment) -> None:
        self.log = exp.log


def consolidate_TRITON_outputs_for_experiment(
    df_scens_finished,
    f_consolidated,
    concat_dims,
    export_format_for_sims,
    compression_level=9,
    compute_attribution=False,
    compare_TRITON_with_TRITONSWMM=False,
    lst_triton_vars_to_drop=None,
    max_mem_usage_gb=0.1,
):
    lst_possible_formats = ["nc", "zarr"]
    if export_format_for_sims not in lst_possible_formats:
        sys.exit(
            f"export format defined as {export_format_for_sims} which is not one of {lst_possible_formats}"
        )
    df_filter = df_scens_finished.model.dropna().str.contains("triton")
    idx_filter = df_filter[df_filter].index
    df_scens_finished_triton = df_scens_finished.loc[idx_filter, :]
    # consolidate triton output netcdfs into single netcdf
    problems = ""
    # load datasets
    dic_triton_outputs_by_source = dict(obs=[], dsgn=[], sim=[])
    # lst_ds_TRITON_out_unprocessed = []
    for idx, scenario_row in df_scens_finished_triton.iterrows():
        name_pattern = f"{scenario_row.model}_{scenario_row.simtype}_yr{int(scenario_row.year)}_{scenario_row.event_type}_{int(scenario_row.event_id)}"
        fname_out_triton = "{}{}.TRITON.{}".format(
            scenario_row.dir_scenario, name_pattern, export_format_for_sims
        )
        data_source = scenario_row.data_source
        if export_format_for_sims == "nc":
            engine = "h5netcdf"
        elif export_format_for_sims == "zarr":
            engine = "zarr"
        try:
            ds = xr.open_dataset(
                fname_out_triton, engine=engine, chunks=dict(timestep_min="auto")
            )
            dic_triton_outputs_by_source[data_source].append(ds)
            # lst_ds_TRITON_out_unprocessed.append(ds)
        except Exception as e:
            problems += f"| Unable to load {fname_out_triton}. It is not included in the {f_consolidated} due to error: {e}|\n"
            continue
    # print("CHECKING TO MAKE SURE THAT THE UNPROCESSED LOADED SIMULATION NETCDFS HAVE MATCHING COMMON DIMENSIONS")
    for key in dic_triton_outputs_by_source.keys():
        lst_ds = dic_triton_outputs_by_source[key]
        if len(lst_ds) > 0:
            df_problems, all_problems = (
                make_sure_lst_of_datasets_have_compatible_dims_for_concatenation(
                    lst_ds, lst_common_dims=["x", "y"]
                )
            )
            if len(all_problems) > 0:
                print(f"DATA SOURCE: {key}")
                print("VALUE COUNTS:")
                print(df_problems.ds_ref.value_counts())  # type: ignore
                print(df_problems.ds_comp.value_counts())  # type: ignore
                print("df_problems")
                print(df_problems)
                problems += all_problems
                print("############################################")
    # lst_ds_TRITON_out = []
    dic_triton_outputs_by_source_processed = dict(obs=[], dsgn=[], sim=[])
    for key in dic_triton_outputs_by_source.keys():
        lst_ds = dic_triton_outputs_by_source[key]
        for ds in lst_ds:
            if len(lst_ds) == 0:
                continue
            added_dims = []
            for concat_dim in concat_dims:
                try:
                    try:
                        dim_value = int(ds.attrs[concat_dim])
                    except:
                        dim_value = str(ds.attrs[concat_dim])
                    # ds.coords[concat_dim] = dim_value
                    ds = ds.assign_coords({concat_dim: dim_value})
                    ds = ds.expand_dims(concat_dim)
                    added_dims.append(concat_dim)
                except:
                    print("Dim {} is missing from the dataset".format(concat_dim))
                    pass
            ds = summarize_triton_simulation_results(ds)
            if lst_triton_vars_to_drop is not None:
                ds = ds.drop_vars(lst_triton_vars_to_drop)
            # at this point in processing, none of the data arrays should have timestep as a coordinate
            timevar_irrelevant = True
            for var in ds.data_vars:
                if "timestep_min" in ds[var].coords:
                    print(
                        "WARNING: {var} still has the timestep_min coordinate for some reason"
                    )
                    timevar_irrelevant = False
            if timevar_irrelevant:
                ds = ds.drop_vars("timestep_min")
            else:
                sys.exit(
                    "Further processing needed to only export non-time-variable simulation summary statistics"
                )
            # lst_ds_TRITON_out.append(ds)
            dic_triton_outputs_by_source_processed[key].append(ds)
    # print("CHECKING TO MAKE SURE THAT THE ***PROCESSED*** SIMULATION NETCDFS HAVE MATCHING COMMON DIMENSIONS")
    for key in dic_triton_outputs_by_source_processed.keys():
        lst_ds = dic_triton_outputs_by_source_processed[key]
        if len(lst_ds) > 0:
            df_problems, all_problems = (
                make_sure_lst_of_datasets_have_compatible_dims_for_concatenation(
                    lst_ds, lst_common_dims=["x", "y"]
                )
            )
            if len(all_problems) > 0:
                print("VALUE COUNTS:")
                print(df_problems.ds_ref.value_counts())  # type: ignore
                print(df_problems.ds_comp.value_counts())  # type: ignore
                print("df_problems")
                print(df_problems)
                problems += all_problems
                print("############################################")
    # combine into 1 dataset dropping attributes that are not identical
    dic_triton_outputs_consolidated = dict()
    for key in dic_triton_outputs_by_source_processed.keys():
        lst_ds = dic_triton_outputs_by_source_processed[key]
        if len(lst_ds) > 0:
            ds_consolidated_outputs = xr.merge(
                lst_ds, combine_attrs="drop_conflicts"
            )  # xr.combine_by_coords(lst_ds_TRITON_out, combine_attrs = "drop_conflicts")
            dic_triton_outputs_consolidated[key] = ds_consolidated_outputs
    # if the total size is less than half a gig, load it into memory; otherwise, rechunk spatial dimension
    dic_problems = dict()
    for key in dic_triton_outputs_consolidated.keys():
        ds_triton_outputs = dic_triton_outputs_consolidated[key]
        f_consolidated_ready_for_writing = f_consolidated
        if key != "sim":
            sim_fname = f_consolidated.split("/")[-1]
            f_consolidated_ready_for_writing = f"{f_consolidated.split(sim_fname)[0]}{sim_fname.split('.' + export_format_for_sims)[0]}_{key}.{export_format_for_sims}"
        memory_size_gb = ds_triton_outputs.nbytes / 1024 / 1024 / 1024
        if memory_size_gb < max_mem_usage_gb:
            print("Loading triton outputs into memory")
            ds_triton_outputs = ds_triton_outputs.load()
        else:
            # print("Rechunking TRITON outputs before further processing and export...")
            ds_triton_outputs = ds_triton_outputs.chunk(dict(x=100, y=100))
        # ds_triton_outputs = xr.concat(lst_ds_TRITON_out, dim = concat_dims, combine_attrs = "drop_conflicts")
        # export
        if compute_attribution:
            print("Computing flood attribution....")
            problems, ds_triton_outputs = compute_tritonswmm_flood_attribution(
                ds_triton_outputs
            )
        if compare_TRITON_with_TRITONSWMM:
            print("Comparing TRITON to TRITON SWMM")
            var_to_compare = "max_wlevel_m"
            comp_model = "triton"
            compare_problems, ds_triton_outputs = compute_diff_with_TRITONSWMM(
                ds_triton_outputs, comp_model=comp_model, var_to_compare=var_to_compare
            )
            problems += compare_problems
        #
        ds_triton_outputs = convert_datavars_to_dtype(
            ds_triton_outputs, lst_dtypes_to_try=[float, str]
        )
        ds_triton_outputs = convert_coords_to_dtype(
            ds_triton_outputs, lst_dtypes_to_try=[int, str]
        )
        if export_format_for_sims == "nc":
            comp = dict(zlib=True, complevel=compression_level)
            encoding = {var: comp for var in ds_triton_outputs.data_vars}
            ds_triton_outputs.to_netcdf(
                f_consolidated_ready_for_writing, encoding=encoding, engine="h5netcdf"
            )
        elif export_format_for_sims == "zarr":
            ds_triton_outputs.to_zarr(
                f_consolidated_ready_for_writing,
                mode="w",
                encoding=return_dic_zarr_encodingds(
                    ds_triton_outputs, clevel=compression_level
                ),
            )
        print(f"finished writing: {f_consolidated_ready_for_writing}")
        dic_problems[key] = problems
    return dic_problems


def make_sure_lst_of_datasets_have_compatible_dims_for_concatenation(
    lst_ds, lst_common_dims=["x", "y"]
):
    # lst_ds = lst_ds_TRITON_out_unprocessed
    all_problems = ""
    lst_s_problems = []
    lst_completed_comparisons = []
    for i, ds_a in enumerate(lst_ds):
        for j, ds_b in enumerate(lst_ds):
            if i == j:
                continue
            # assign the ds_ref and ds_comp based on alphabetical order of source files
            lst_files = [ds_a.encoding["source"], ds_b.encoding["source"]]
            lst_files.sort()
            if ds_a.encoding["source"] == lst_files[0]:
                ds_ref = ds_a
                ds_comp = ds_b
            elif ds_b.encoding["source"] == lst_files[0]:
                ds_ref = ds_b
                ds_comp = ds_a
            else:
                sys.exit("SOMETHING CRAZY")
            f_ref = ds_ref.encoding["source"]
            f_comp = ds_comp.encoding["source"]
            # compare commone dimensions to make sure they match
            if lst_files not in lst_completed_comparisons:
                s_problems = pd.Series(index=["ds_ref", "ds_comp", "problems"]).astype(
                    str
                )
                all_problems = make_sure_ds_are_compatible_for_concatenation(
                    ds_ref, ds_comp, lst_common_dims
                )
                lst_completed_comparisons.append(lst_files)
                s_problems.loc["ds_ref"] = f_ref
                s_problems.loc["ds_comp"] = f_comp
                s_problems.loc["problems"] = all_problems
                lst_s_problems.append(s_problems)
    if len(lst_s_problems) == 1:
        df_problems = lst_s_problems[0].to_frame().T
        df_problems = df_problems[df_problems["problems"] != ""].reset_index()
    elif len(lst_s_problems) > 1:
        df_problems = pd.concat(lst_s_problems, axis=1).T
        df_problems = df_problems[df_problems["problems"] != ""].reset_index()
    else:
        df_problems = None
    return df_problems, all_problems


def make_sure_ds_are_compatible_for_concatenation(
    ds_ref, ds_comp, lst_common_dims=["x", "y"]
):
    all_problems = ""
    problems = check_matching_dimensions(ds_ref, ds_comp)
    matching_dim_problems = check_for_matching_dim_values(
        ds_ref, ds_comp, lst_common_dims
    )
    all_problems += problems + matching_dim_problems
    # print(all_problems)
    return all_problems


def check_matching_dimensions(ds_ref, ds_comp):
    problems = ""
    lst_common_dims = []
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in ds_ref.dims:
        if dim not in ds_comp.dims:
            problems += f"| WARNING: {dim} in {f_ref} but not in {f_comp} |\n"
        else:
            lst_common_dims.append(dim)
            # print(problems)
    for dim in ds_comp.dims:
        if dim not in ds_ref.dims:
            problems += f"| WARNING: {dim} in {f_comp} but not in {f_ref} |\n"
    # print(problems)
    return problems


def check_for_matching_dim_values(ds_ref, ds_comp, lst_common_dims=["x", "y"]):
    problems = ""
    f_ref = ds_ref.encoding["source"]
    f_comp = ds_comp.encoding["source"]
    for dim in lst_common_dims:
        ar_dif = ds_ref[dim].values - ds_comp[dim].values
        n_diff = ((ar_dif) != 0).sum()
        if n_diff > 0:
            problems += (
                f"| WARNING: {dim} values are not all equal in {f_ref} and {f_comp} |\n"
            )
    # print(problems)
    return problems


def summarize_triton_simulation_results(ds):
    # compute max velocity, time of max velocity, and the x and y components of the max velocity
    ds["velocity_mps"] = (ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5
    ## compute max velocity
    ds["max_velocity_mps"] = ds["velocity_mps"].max(dim="timestep_min", skipna=True)
    ## compute time of max velocity
    ds["time_of_max_velocity_min"] = ds["velocity_mps"].idxmax(
        dim="timestep_min", skipna=True
    )
    ## return x and y velocities at time of max velocity
    ds["velocity_x_mps_at_time_of_max_velocity"] = ds["velocity_x_mps"].sel(
        timestep_min=ds["time_of_max_velocity_min"]
    )
    ds["velocity_y_mps_at_time_of_max_velocity"] = ds["velocity_y_mps"].sel(
        timestep_min=ds["time_of_max_velocity_min"]
    )
    ## where the max velocity is zero, fill with np.nan
    ds["time_of_max_velocity_min"] = xr.where(
        ds["max_velocity_mps"] == 0, np.nan, ds["time_of_max_velocity_min"]
    )
    ds["velocity_x_mps_at_time_of_max_velocity"] = xr.where(
        ds["max_velocity_mps"] == 0,
        np.nan,
        ds["velocity_x_mps_at_time_of_max_velocity"],
    )
    ds["velocity_y_mps_at_time_of_max_velocity"] = xr.where(
        ds["max_velocity_mps"] == 0,
        np.nan,
        ds["velocity_y_mps_at_time_of_max_velocity"],
    )
    ## drop velocity_mps
    ds = ds.drop_vars("velocity_mps")
    ############################################
    # compute max water level and time of max water level
    if "timestep_min" in ds.max_wlevel_m.dims:
        ds["max_wlevel_m"] = ds.max_wlevel_m.sel(
            timestep_min=ds.max_wlevel_m.timestep_min.to_series().max()
        ).reset_coords(drop=True)
    ds["time_of_max_wlevel_min"] = ds["wlevel_m"].idxmax(
        dim="timestep_min", skipna=True
    )
    ## where the max wlevel is zero, fill with np.nan
    ds["time_of_max_wlevel_min"] = xr.where(
        ds["max_wlevel_m"] == 0, np.nan, ds["time_of_max_wlevel_min"]
    )
    return ds


def compute_tritonswmm_flood_attribution(ds_all_outputs, model="tritonswmm"):
    # ds_all_outputs = ds_triton_outputs
    # compute flood attribution by gridcell
    ds_triton = ds_all_outputs.sel(model=model)
    problems = ""
    lst_ds_interactions = []
    # identify unique coordinates to loop through
    lst_dic_storm_sel = return_lst_dic_of_unique_storm_idxs(ds=ds_triton)
    #
    for dict_sel in lst_dic_storm_sel:
        ds_event = ds_triton.sel(dict_sel)
        ds_interaction = xr.Dataset()
        # select the last timestep with non null outputs
        ## first subset non-null outputs
        da_event_maxwlevel = ds_event["max_wlevel_m"]
        # if timestep is present in coordinates of max wlevel, select the last non-null timestep
        if "timestep_min" in da_event_maxwlevel.coords:
            da_event_maxwlevel = da_event_maxwlevel.where(
                ~da_event_maxwlevel.isnull().compute(), drop=True
            )
            ## select last timestep corresponding to max water level in the entire simulation
            da_event_maxwlevel = da_event_maxwlevel.sel(
                timestep_min=da_event_maxwlevel["timestep_min"].values.max()
            )
        # drop where all values are null
        mask_null = (
            (da_event_maxwlevel.sel(simtype="compound").isnull())
            | (da_event_maxwlevel.sel(simtype="rainonly").isnull())
            | (da_event_maxwlevel.sel(simtype="surgeonly").isnull())
        )
        mask_allvalid = ~mask_null
        # Expand the mask to have the same dimensions as the original DataArray
        mask_allvalid = mask_allvalid.broadcast_like(da_event_maxwlevel)
        da_event_maxwlevel = da_event_maxwlevel.where(
            mask_allvalid.compute(), drop=True
        )
        da_shape = da_event_maxwlevel.shape
        nvals = 1
        for dim in da_shape:
            nvals *= dim
        # if values are missing, halt the loop and record the problem
        if nvals == 0:
            prnt_strng = f"| Warning: there are no valid max water level simulation outputs for event ({dict_sel})|\n"
            print(prnt_strng)
            problems = problems + prnt_strng
            continue
        # compute attribution (suppressing divide by zero warnings)
        with np.errstate(divide="ignore", invalid="ignore"):
            da_event_sum_rainandsurge = da_event_maxwlevel.sel(
                simtype="surgeonly"
            ) + da_event_maxwlevel.sel(simtype="rainonly")
            da_event_frac_rain = da_event_maxwlevel.sel(
                simtype="rainonly"
            ) / da_event_maxwlevel.sel(simtype="compound")
            da_event_frac_surge = da_event_maxwlevel.sel(
                simtype="surgeonly"
            ) / da_event_maxwlevel.sel(simtype="compound")
            da_event_frac_interaction = (
                da_event_maxwlevel.sel(simtype="compound") - da_event_sum_rainandsurge
            ) / da_event_maxwlevel.sel(simtype="compound")
            # where the driver of interest has zero flooding, assign an attribution of zero
            da_event_frac_rain = xr.where(
                (da_event_maxwlevel.sel(simtype="rainonly") == 0), 0, da_event_frac_rain
            )
            da_event_frac_surge = xr.where(
                (da_event_maxwlevel.sel(simtype="surgeonly") == 0),
                0,
                da_event_frac_surge,
            )
            da_event_frac_interaction = xr.where(
                (da_event_maxwlevel.sel(simtype="compound") == 0),
                0,
                da_event_frac_interaction,
            )
            # if rain only is non zero and compound is zero, assign infinity
            da_event_frac_rain = xr.where(
                (da_event_maxwlevel.sel(simtype="rainonly") != 0)
                & (da_event_maxwlevel.sel(simtype="compound") == 0),
                np.inf,
                da_event_frac_rain,
            )
            # if surge only is non zero and compound is zero, assign infinity
            da_event_frac_surge = xr.where(
                (da_event_maxwlevel.sel(simtype="surgeonly") != 0)
                & (da_event_maxwlevel.sel(simtype="compound") == 0),
                np.inf,
                da_event_frac_surge,
            )
            # check if there are still any missing values in the frac interactions
            if (
                check_da_for_na(da_event_frac_rain)
                or check_da_for_na(da_event_frac_surge)
                or check_da_for_na(da_event_frac_interaction)
            ):
                prnt_strng = f"| Warning: an attribution data array contains missing values for event ({dict_sel}) |\n"
                print(prnt_strng)
                problems = problems + prnt_strng
            # check to make sure that if there is non-zero flooding in the surgeonly OR rainonly model, there is flooding in the compound model
            ## where this is the case, i assigned a value of infinity so I can just count n values where frac = inf
            n_invalid = (da_event_frac_rain == np.inf).values.sum() + (
                da_event_frac_surge == np.inf
            ).values.sum()
            if n_invalid > 0:
                prnt_strng = f"| WARNING: There are {n_invalid} gridcells in event ({dict_sel}) where there is flooding in the surge only or rain only simulations but NOT in the compound simulations. These simulations need to be inspected. The interaction value is assigned to infinity in these cases.|\n"
                print(prnt_strng)
                problems = problems + prnt_strng
            # add attribution data to the netcdf
            ds_interaction["frac_rain"] = da_event_frac_rain
            ds_interaction["frac_surge"] = da_event_frac_surge
            ds_interaction["frac_interaction"] = da_event_frac_interaction
            for da_name in ds_interaction.data_vars:
                for dim in ds_triton.dims:
                    # print(dim)
                    if (dim not in ds_interaction[da_name].dims) and (
                        dim not in ["timestep_min", "model", "simtype"]
                    ):
                        ds_interaction[da_name] = ds_interaction[da_name].expand_dims(
                            {dim: [pd.Series(ds_interaction[dim].values).iloc[0]]}
                        )
                        # ds_t_to_ts_compare[da_name] = ds_t_to_ts_compare[da_name].expand_dims({dim:pd.Series(ds_t_to_ts_compare[dim].values)})
            # append to list
            lst_ds_interactions.append(ds_interaction)
    ds_attribution = xr.combine_by_coords(lst_ds_interactions).reset_coords(drop=True)
    ds_all_outputs = xr.merge([ds_attribution, ds_all_outputs])
    print("finished computing flood attribution")
    return problems, ds_all_outputs


def check_da_for_na(da):
    # Check for NaN values
    nan_mask = da.isnull()
    # Check if any NaN values are present
    any_nans = bool(nan_mask.any().values)
    return any_nans


def return_lst_dic_of_unique_storm_idxs(ds):
    lst_coords = []
    for coord in ds.coords:
        if coord not in [
            "x",
            "y",
            "model",
            "simtype",
            "link_id",
            "node_id",
        ]:  # and (len(ds_triton[coord].values)>1):
            lst_coords.append(coord)
    # find unique indices for unique storm ids
    if "max_wlevel_m" in ds.data_vars:
        datavar = "max_wlevel_m"
        idx_loc = dict(x=1, y=1)
    elif "max_flow_cms" in ds.data_vars:
        datavar = "max_flow_cms"
        idx_loc = dict(link_id=1)
    elif "total_inflow_vol_10e6_ltr" in ds.data_vars:
        datavar = "total_inflow_vol_10e6_ltr"
        idx_loc = dict(node_id=1)
    if "x" in ds.coords and "y" in ds.coords:
        idx_storms = (
            ds.isel(idx_loc)[datavar]
            .to_dataframe()
            .reset_index()
            .set_index(lst_coords)
            .index.unique()
        )
    else:
        idx_storms = (
            ds.isel(idx_loc)[datavar]
            .to_dataframe()
            .reset_index()
            .set_index(lst_coords)
            .index.unique()
        )
    idx_names = idx_storms.names
    lst_dic_storm_sel = []
    for idx in idx_storms:
        dic_sel = dict()
        for i, name in enumerate(idx_names):
            dic_sel[name] = idx[i]
        lst_dic_storm_sel.append(dic_sel)
    return lst_dic_storm_sel


def compute_diff_with_TRITONSWMM(ds_all_outputs, comp_model, var_to_compare):
    # extract compound results
    # ds_all_outputs = ds_triton_outputs
    # ds_all_outputs = ds_combined_nodes, comp_model="swmm", var_to_compare = "flooding_cms"
    ds_compound = ds_all_outputs.sel(simtype="compound")
    problems = ""
    lst_ds_ts_to_t_compare = []
    lst_dic_storm_sel = return_lst_dic_of_unique_storm_idxs(ds=ds_compound)
    for dict_sel in lst_dic_storm_sel:
        # for year in ds_compound["year"].values:
        #     for event_type in ds_compound["event_type"].values:
        #         for event_id in ds_compound["event_id"].values:
        ds_ts_to_other_compare = xr.Dataset()
        # da_event = ds_compound.sel(realization = rz, storm_id = strm, year = yr)[var_to_compare]
        da_event = ds_compound.sel(dict_sel)[var_to_compare]
        # da_event = subset_weather_event(ds=ds_compound, year=year, event_type=event_type, event_id=event_id)[var_to_compare]
        # select the last timestep with non null outputs
        ## first subset non-null outputs
        da_event = da_event.where(~da_event.isnull().compute(), drop=True)
        if "tritonswmm" in da_event.model.values:
            pass
        else:
            problem_text = f"| Warning: there are no valid {var_to_compare} tritonswmm simulation outputs for event ({dict_sel}) |\n"
            print(problem_text)
            problems += problem_text
            continue
        ## select last timestep corresponding to max water level in the entire simulation
        if "timestep_min" in da_event.coords:
            da_event = da_event.sel(timestep_min=da_event["timestep_min"].values.max())
        if "date_time" in da_event.coords:
            sys.exit(
                f"Script attempting to compre tritonswmm to {comp_model} {var_to_compare}. I have not developed this function to handle this SWMM output."
            )
        ## drop where all values are null
        if comp_model not in da_event.model.values:
            print(
                f"WARNING: {comp_model} outputs missing for year, event_type, event_id {dict_sel}. Skipping processing of this simulation."
            )
            continue
        mask_null = (da_event.sel(model=comp_model).isnull()) | (
            da_event.sel(model="tritonswmm").isnull()
        )
        mask_allvalid = ~mask_null
        ## Expand the mask to have the same dimensions as the original DataArray
        mask_allvalid = mask_allvalid.broadcast_like(da_event)
        da_event = da_event.where(mask_allvalid.compute(), drop=True)
        da_shape = da_event.shape
        nvals = 1
        for dim in da_shape:
            nvals *= dim
        ## if there are no valid values, continue the loop and record the problem
        if nvals == 0:
            prnt_strng = f"| Warning: there are no valid {var_to_compare} simulation outputs for event ({dict_sel}) |\n"
            print(prnt_strng)
            problems = problems + prnt_strng
            continue
        # computing difference between triton results and other model results
        da_event_other = da_event.sel(model=comp_model)
        da_event_ts = da_event.sel(model="tritonswmm")
        da_ts_minus_other = da_event_ts - da_event_other
        da_ts_to_other_perc_diff = da_ts_minus_other / da_event_ts
        # where both the triton and triton swmm result is zero, assign a percent difference of zero
        da_ts_to_other_perc_diff = xr.where(
            (da_event_ts == 0) & (da_event_other == 0), 0, da_ts_to_other_perc_diff
        )
        # in the case where the triton result is nonzero and the tritonswmm result is zero, assign infinity
        mask_other_nonzero_tritonswmm_zero = (da_event_ts == 0) & (da_event_other != 0)
        da_ts_to_other_perc_diff = xr.where(
            mask_other_nonzero_tritonswmm_zero, np.inf, da_ts_to_other_perc_diff
        )
        # (da_ts_to_other_perc_diff == np.inf).values.sum()
        # add data array
        ds_ts_to_other_compare[f"{var_to_compare}_tritonswmm_minus_{comp_model}"] = (
            da_ts_minus_other
        )
        ds_ts_to_other_compare[
            f"{var_to_compare}_tritonswmm_to_{comp_model}_percdiff"
        ] = da_ts_to_other_perc_diff
        # add dimensions for combining
        for da_name in ds_ts_to_other_compare.data_vars:
            for dim in ds_all_outputs.dims:
                # print(dim)
                if (dim not in ds_ts_to_other_compare[da_name].dims) and (
                    dim not in ["date_time", "timestep_min", "model", "simtype"]
                ):
                    ds_ts_to_other_compare[da_name] = ds_ts_to_other_compare[
                        da_name
                    ].expand_dims(
                        {dim: [pd.Series(ds_ts_to_other_compare[dim].values).iloc[0]]}
                    )
        # append to list
        lst_ds_ts_to_t_compare.append(ds_ts_to_other_compare)
    ds_comparison = xr.combine_by_coords(lst_ds_ts_to_t_compare).reset_coords(drop=True)
    ds_all_outputs = xr.merge([ds_comparison, ds_all_outputs])
    return problems, ds_all_outputs
