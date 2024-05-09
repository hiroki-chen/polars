#[cfg(feature = "csv")]
mod csv;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;

#[cfg(feature = "ipc")]
mod support;
use std::mem;
#[cfg(any(feature = "parquet", feature = "ipc", feature = "cse"))]
use std::ops::Deref;

#[cfg(feature = "csv")]
pub(crate) use csv::CsvExec;
#[cfg(feature = "ipc")]
pub(crate) use ipc::IpcExec;
#[cfg(feature = "parquet")]
pub(crate) use parquet::ParquetExec;
use picachv::get_data_argument::DataSource;
use picachv::{plan_argument, GetDataArgument, GetDataInMemory, PlanArgument};
#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars_io::predicates::PhysicalIoExpr;
#[cfg(any(feature = "parquet", feature = "csv", feature = "ipc", feature = "cse"))]
use polars_io::prelude::*;
use polars_plan::global::_set_n_rows_for_scan;
#[cfg(feature = "ipc")]
pub(crate) use support::ConsecutiveCountState;

use super::*;
#[cfg(any(feature = "ipc", feature = "parquet"))]
use crate::physical_plan::expressions::phys_expr_to_io_expr;
use crate::prelude::*;

#[cfg(any(feature = "ipc", feature = "parquet"))]
type Projection = Option<Vec<usize>>;
#[cfg(any(feature = "ipc", feature = "parquet"))]
type Predicate = Option<Arc<dyn PhysicalIoExpr>>;

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn prepare_scan_args(
    predicate: Option<Arc<dyn PhysicalExpr>>,
    with_columns: &mut Option<Arc<Vec<String>>>,
    schema: &mut SchemaRef,
    has_row_index: bool,
    hive_partitions: Option<&[Series]>,
) -> (Projection, Predicate) {
    let with_columns = mem::take(with_columns);
    let schema = mem::take(schema);

    let projection = materialize_projection(
        with_columns.as_deref().map(|cols| cols.deref()),
        &schema,
        hive_partitions,
        has_row_index,
    );

    let predicate = predicate.map(phys_expr_to_io_expr);

    (projection, predicate)
}

/// Producer of an in memory DataFrame
pub struct DataFrameExec {
    pub(crate) df: Arc<DataFrame>,
    pub(crate) selection: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) projection: Option<Arc<Vec<String>>>,
    pub(crate) predicate_has_windows: bool,
}

impl DataFrameExec {
    pub fn new(
        df: Arc<DataFrame>,
        selection: Option<Arc<dyn PhysicalExpr>>,
        projection: Option<Arc<Vec<String>>>,
        predicate_has_windows: bool,
    ) -> Self {
        DataFrameExec {
            df,
            selection,
            projection,
            predicate_has_windows,
        }
    }
}

impl Executor for DataFrameExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let uuid = self.df.get_uuid();
        state.set_active_df_uuid(uuid);

        let df = mem::take(&mut self.df);
        let mut df = Arc::try_unwrap(df).unwrap_or_else(|df| (*df).clone());

        // projection should be before selection as those are free
        // TODO: this is only the case if we don't create new columns
        if let Some(projection) = &self.projection {
            df = df.select(projection.as_ref())?;
        }

        // TODO: Do this.
        if let Some(selection) = &self.selection {
            if self.predicate_has_windows {
                state.insert_has_window_function_flag()
            }
            let s = selection.evaluate(&df, state)?;
            if self.predicate_has_windows {
                state.clear_window_expr_cache()
            }
            let mask = s.bool().map_err(
                |_| polars_err!(ComputeError: "filter predicate was not of type boolean"),
            )?;
            let pred_bool = mask
                .iter()
                .map(|b| {
                    b.ok_or(polars_err!(ComputeError: "filter predicate was not of type boolean"))
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            state.transform.push_filter(uuid, &pred_bool).map_err(
                |e| polars_err!(ComputeError: format!("Could not push filter transform: {}", e)),
            )?;
            df = df.filter(mask)?;
        }

        let df = match _set_n_rows_for_scan(None) {
            Some(limit) => df.head(Some(limit)),
            None => df,
        };

        let plan_arg = PlanArgument {
            argument: Some(plan_argument::Argument::GetData(GetDataArgument {
                data_source: Some(DataSource::InMemory(GetDataInMemory {
                    df_uuid: state.active_df_uuid.clone().to_bytes_le().to_vec(),
                    pred: self
                        .selection
                        .as_ref()
                        .map(|e| e.get_uuid().to_bytes_le().to_vec()),
                    projected_list: self
                        .projection
                        .as_ref()
                        .cloned()
                        .unwrap_or_default()
                        .iter()
                        .map(|s| s.clone())
                        .collect::<Vec<_>>(),
                })),
            })),
        };
        self.execute_epilogue(state, Some(plan_arg))?;

        println!("after scan: uuid = {}", state.active_df_uuid);
        Ok(df)
    }
}

pub(crate) struct AnonymousScanExec {
    pub(crate) function: Arc<dyn AnonymousScan>,
    pub(crate) file_options: FileScanOptions,
    pub(crate) file_info: FileInfo,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) output_schema: Option<SchemaRef>,
    pub(crate) predicate_has_windows: bool,
}

impl Executor for AnonymousScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let mut args = AnonymousScanArgs {
            n_rows: self.file_options.n_rows,
            with_columns: self.file_options.with_columns.clone(),
            schema: self.file_info.schema.clone(),
            output_schema: self.output_schema.clone(),
            predicate: None,
        };
        if self.predicate.is_some() {
            state.insert_has_window_function_flag()
        }

        match (self.function.allows_predicate_pushdown(), &self.predicate) {
            (true, Some(predicate)) => state.record(
                || {
                    args.predicate = predicate.as_expression().cloned();
                    self.function.scan(args)
                },
                "anonymous_scan".into(),
            ),
            (false, Some(predicate)) => state.record(
                || {
                    let mut df = self.function.scan(args)?;
                    let s = predicate.evaluate(&df, state)?;
                    if self.predicate_has_windows {
                        state.clear_window_expr_cache()
                    }
                    let mask = s.bool().map_err(
                        |_| polars_err!(ComputeError: "filter predicate was not of type boolean"),
                    )?;
                    df = df.filter(mask)?;

                    Ok(df)
                },
                "anonymous_scan".into(),
            ),
            _ => state.record(|| self.function.scan(args), "anonymous_scan".into()),
        }
    }
}
