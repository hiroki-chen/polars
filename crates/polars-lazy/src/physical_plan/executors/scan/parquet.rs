use std::path::PathBuf;

use picachv::native::{register_policy_dataframe_bin, register_policy_dataframe_parquet};
use picachv::transform_info::Information;
use picachv::FilterInformation;
use polars_core::config;
#[cfg(feature = "cloud")]
use polars_core::config::{get_file_prefetch_size, verbose};
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::cloud::CloudOptions;
use polars_io::{is_cloud_url, RowIndex};
use uuid::Uuid;

use super::*;

pub struct ParquetExec {
    paths: Arc<[PathBuf]>,
    file_info: FileInfo,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    #[allow(dead_code)]
    metadata: Option<Arc<FileMetaData>>,
    with_policy: Option<Arc<PathBuf>>,
    active_df_uuid: Option<Uuid>,
}

impl ParquetExec {
    pub(crate) fn load_policy(&mut self, df: &DataFrame, mask: Option<ChunkedArray<BooleanType>>, ctx_id: Uuid) -> PolarsResult<()> {
        // HACK: This is to reliably exclude the time taken to register the policy.
        if let Some(policy) = self.with_policy.as_ref() {
            let filter = mask.as_ref().map(|mask| mask.iter().map(|e| e.unwrap_or_default()).collect::<Vec<_>>());
            let projection = match materialize_projection(self.file_options.with_columns.as_ref().map(|e| e.as_slice()), &self.file_info.schema, None, false) {
                Some(proj) => proj,
                None => df.fields().iter().enumerate().map(|(i, _)| i as usize).collect::<Vec<_>>(),
            };

            self.active_df_uuid.replace(
            register_policy_dataframe_parquet(ctx_id, self.with_policy.as_ref().unwrap().as_path().to_str().unwrap(), &projection, filter.as_ref().map(|f| f.as_slice())).map_err(|e| PolarsError::InvalidOperation(e.to_string().into()))?);
        }

        Ok(())
    }

    pub(crate) fn new(
        paths: Arc<[PathBuf]>,
        file_info: FileInfo,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        metadata: Option<Arc<FileMetaData>>,
        with_policy: Option<Arc<PathBuf>>,
        ctx_id: Uuid,
    ) -> PolarsResult<Self> {
        let mut executor = ParquetExec {
            paths,
            file_info,
            predicate,
            options,
            cloud_options,
            file_options,
            metadata,
            with_policy,
            active_df_uuid: None,
        };

        if let Some(policy) = executor.with_policy.as_ref() {
            let (df, mask) = executor.read()?;
            executor.load_policy(&df, mask, ctx_id)?;
        }

        Ok(executor)
    }

    fn read_par(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let parallel = match self.options.parallel {
            ParallelStrategy::Auto if self.paths.len() > POOL.current_num_threads() => {
                ParallelStrategy::RowGroups
            },
            identity => identity,
        };

        let mut result = vec![];

        let mut remaining_rows_to_read = self.file_options.n_rows.unwrap_or(usize::MAX);
        let mut base_row_index = self.file_options.row_index.take();

        // Limit no. of files at a time to prevent open file limits.
        for paths in self
            .paths
            .chunks(std::cmp::min(POOL.current_num_threads(), 128))
        {
            if remaining_rows_to_read == 0 && !result.is_empty() {
                return Ok(result);
            }

            // First initialize the readers, predicates and metadata.
            // This will be used to determine the slices. That way we can actually read all the
            // files in parallel even if we add row index columns or slices.
            let readers_and_metadata = paths
                .iter()
                .map(|path| {
                    let mut file_info = self.file_info.clone();
                    file_info.update_hive_partitions(path)?;

                    let hive_partitions = file_info
                        .hive_parts
                        .as_ref()
                        .map(|hive| hive.materialize_partition_columns());

                    let file = std::fs::File::open(path)?;
                    let (projection, predicate) = prepare_scan_args(
                        self.predicate.clone(),
                        &mut self.file_options.with_columns.clone(),
                        &mut self.file_info.schema.clone(),
                        base_row_index.is_some(),
                        hive_partitions.as_deref(),
                    );

                    let mut reader = ParquetReader::new(file)
                        .with_schema(self.file_info.reader_schema.clone())
                        .read_parallel(parallel)
                        .set_low_memory(self.options.low_memory)
                        .use_statistics(self.options.use_statistics)
                        .set_rechunk(false)
                        .with_hive_partition_columns(hive_partitions);

                    reader
                        .num_rows()
                        .map(|num_rows| (reader, num_rows, predicate, projection))
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            let iter = readers_and_metadata
                .iter()
                .map(|(_, num_rows, _, _)| *num_rows);

            let rows_statistics = get_sequential_row_statistics(iter, remaining_rows_to_read);

            let out = POOL.install(|| {
                readers_and_metadata
                    .into_par_iter()
                    .zip(rows_statistics.par_iter())
                    .map(
                        |(
                            (reader, num_rows_this_file, predicate, projection),
                            (remaining_rows_to_read, cumulative_read),
                        )| {
                            let remaining_rows_to_read = *remaining_rows_to_read;
                            let remaining_rows_to_read =
                                if num_rows_this_file < remaining_rows_to_read {
                                    None
                                } else {
                                    Some(remaining_rows_to_read)
                                };
                            let row_index = base_row_index.as_ref().map(|rc| RowIndex {
                                name: rc.name.clone(),
                                offset: rc.offset + *cumulative_read as IdxSize,
                            });

                            reader
                                .with_n_rows(remaining_rows_to_read)
                                .with_row_index(row_index)
                                // .with_predicate(predicate.clone())
                                .with_projection(projection.clone())
                                .finish()
                        },
                    )
                    .collect::<PolarsResult<Vec<_>>>()
            })?;

            let n_read = out.iter().map(|df| df.height()).sum();
            remaining_rows_to_read = remaining_rows_to_read.saturating_sub(n_read);
            if let Some(rc) = &mut base_row_index {
                rc.offset += n_read as IdxSize;
            }
            if result.is_empty() {
                result = out;
            } else {
                result.extend_from_slice(&out)
            }
        }
        Ok(result)
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let verbose = verbose();
        let first_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .expect("should be set");
        let first_metadata = &self.metadata;
        let cloud_options = self.cloud_options.as_ref();
        let with_columns = self
            .file_options
            .with_columns
            .as_ref()
            .map(|v| v.as_slice());

        let mut result = vec![];
        let batch_size = get_file_prefetch_size();

        if verbose {
            eprintln!("POLARS PREFETCH_SIZE: {}", batch_size)
        }

        let mut remaining_rows_to_read = self.file_options.n_rows.unwrap_or(usize::MAX);
        let mut base_row_index = self.file_options.row_index.take();
        let mut processed = 0;
        for (batch_idx, paths) in self.paths.chunks(batch_size).enumerate() {
            if remaining_rows_to_read == 0 && !result.is_empty() {
                return Ok(result);
            }
            processed += paths.len();
            if verbose {
                eprintln!(
                    "querying metadata of {}/{} files...",
                    processed,
                    self.paths.len()
                );
            }

            // First initialize the readers and get the metadata concurrently.
            let iter = paths.iter().enumerate().map(|(i, path)| async move {
                let first_file = batch_idx == 0 && i == 0;
                // use the cached one as this saves a cloud call
                let (metadata, schema) = if first_file {
                    (first_metadata.clone(), Some((*first_schema).clone()))
                } else {
                    (None, None)
                };
                let mut reader = ParquetAsyncReader::from_uri(
                    &path.to_string_lossy(),
                    cloud_options,
                    // Schema must be the same for all files. The hive partitions are included in this schema.
                    schema,
                    metadata,
                )
                .await?;

                if !first_file {
                    let schema = reader.schema().await?;
                    check_projected_arrow_schema(
                        first_schema.as_ref(),
                        schema.as_ref(),
                        with_columns,
                        "schema of all files in a single scan_parquet must be equal",
                    )?
                }

                let num_rows = reader.num_rows().await?;
                PolarsResult::Ok((num_rows, reader))
            });
            let readers_and_metadata = futures::future::try_join_all(iter).await?;

            // Then compute `n_rows` to be taken per file up front, so we can actually read concurrently
            // after this.
            let iter = readers_and_metadata
                .iter()
                .map(|(num_rows, _)| num_rows)
                .copied();

            let rows_statistics = get_sequential_row_statistics(iter, remaining_rows_to_read);

            // Now read the actual data.
            let file_info = &self.file_info;
            let file_options = &self.file_options;
            let use_statistics = self.options.use_statistics;
            let predicate = &self.predicate;
            let base_row_index_ref = &base_row_index;

            if verbose {
                eprintln!("reading of {}/{} file...", processed, self.paths.len());
            }

            let iter = readers_and_metadata
                .into_iter()
                .zip(rows_statistics.iter())
                .zip(paths.as_ref().iter())
                .map(
                    |(
                        ((num_rows_this_file, reader), (remaining_rows_to_read, cumulative_read)),
                        path,
                    )| async move {
                        let mut file_info = file_info.clone();
                        let remaining_rows_to_read = *remaining_rows_to_read;
                        let remaining_rows_to_read = if num_rows_this_file < remaining_rows_to_read
                        {
                            None
                        } else {
                            Some(remaining_rows_to_read)
                        };
                        let row_index = base_row_index_ref.as_ref().map(|rc| RowIndex {
                            name: rc.name.clone(),
                            offset: rc.offset + *cumulative_read as IdxSize,
                        });

                        file_info.update_hive_partitions(path)?;

                        let hive_partitions = file_info
                            .hive_parts
                            .as_ref()
                            .map(|hive| hive.materialize_partition_columns());

                        let (projection, predicate) = prepare_scan_args(
                            predicate.clone(),
                            &mut file_options.with_columns.clone(),
                            &mut file_info.schema.clone(),
                            row_index.is_some(),
                            hive_partitions.as_deref(),
                        );

                        reader
                            .with_n_rows(remaining_rows_to_read)
                            .with_row_index(row_index)
                            .with_projection(projection)
                            .use_statistics(use_statistics)
                            .with_predicate(predicate)
                            .set_rechunk(false)
                            .with_hive_partition_columns(hive_partitions)
                            .finish()
                            .await
                            .map(Some)
                    },
                );

            let dfs = futures::future::try_join_all(iter).await?;
            let n_read = dfs
                .iter()
                .map(|opt_df| opt_df.as_ref().map(|df| df.height()).unwrap_or(0))
                .sum();
            remaining_rows_to_read = remaining_rows_to_read.saturating_sub(n_read);
            if let Some(rc) = &mut base_row_index {
                rc.offset += n_read as IdxSize;
            }
            result.extend(dfs.into_iter().flatten())
        }

        Ok(result)
    }

    fn read(&mut self) -> PolarsResult<(DataFrame, Option<BooleanChunked>)> {
        // FIXME: The row index implementation is incorrect when a predicate is
        // applied. This code mitigates that by applying the predicate after the
        // collection of the entire dataframe if a row index is requested. This is
        // inefficient.
        // let post_predicate = self
        //     .file_options
        //     .row_index
        //     .as_ref()
        //     .and_then(|_| self.predicate.take())
        //     .map(phys_expr_to_io_expr);
        let post_predicate = self.predicate.clone().map(phys_expr_to_io_expr);

        let is_cloud = match self.paths.first() {
            Some(p) => is_cloud_url(p.as_path()),
            None => {
                let hive_partitions = self
                    .file_info
                    .hive_parts
                    .as_ref()
                    .map(|hive| hive.materialize_partition_columns());
                let (projection, _) = prepare_scan_args(
                    None,
                    &mut self.file_options.with_columns,
                    &mut self.file_info.schema,
                    self.file_options.row_index.is_some(),
                    hive_partitions.as_deref(),
                );
                return Ok((materialize_empty_df(
                    projection.as_deref(),
                    self.file_info.reader_schema.as_ref().unwrap(),
                    hive_partitions.as_deref(),
                    self.file_options.row_index.as_ref(),
                ), None));
            },
        };
        let force_async = config::force_async();

        let out = if is_cloud || force_async {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if !is_cloud && config::verbose() {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime().block_on_potential_spawn(self.read_async())?
            }
        } else {
            self.read_par()?
        };

        let mut out = accumulate_dataframes_vertical(out)?;
        let mask = match post_predicate.as_ref() {
            Some(mask) => Some(mask.evaluate_io(&out)?.bool()?.clone()),
            None => None,
        };
        polars_io::predicates::apply_predicate(&mut out, post_predicate.as_deref(), true)?;

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }
        Ok((out, mask))
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        polars_ensure!(!(state.policy_check && self.with_policy.is_none()), 
            InvalidOperation: "Policy check requested but no policy was provided");

        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.paths[0].to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("parquet".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        let (df, mask) = state.record(|| self.read(), profile_name)?;

        if state.policy_check {
            if self.active_df_uuid.is_none() {
                self.load_policy(&df, mask, state.ctx_id)?;
            }

            state.active_df_uuid = self.active_df_uuid.clone().unwrap();

            let project_list = match self.file_options.with_columns.as_ref() {
                    Some(project_list) => {
                        let project_list = project_list
                            .iter()
                            .map(|s| {
                                df.get_column_index(s)
                                    .ok_or(PolarsError::ComputeError(
                                        format!("Column {} not found", s).into(),
                                    ))
                                    .map(|e| e as u64)
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;
    
                        Some(ProjectList { project_list })
                    },
                    None => None,
                };

            // Then construct the plan argument and execute the epilogue.
            let plan_arg = PlanArgument {
                argument: Some(plan_argument::Argument::GetData(GetDataArgument {
                    data_source: Some(DataSource::InMemory(GetDataInMemory {
                        df_uuid: state.active_df_uuid.clone().to_bytes_le().to_vec(),
                        pred: None,
                        project_list,
                    })),
                })),
                transform_info: state.transform.clone(),
            };

            self.execute_epilogue(state, Some(plan_arg))?;
        }

        Ok(df)
    }
}

