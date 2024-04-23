use picachv::native::{execute_epilogue, execute_prologue};
use uuid::Uuid;

use super::*;

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send {
    fn execute(&mut self, cache: &mut ExecutionState) -> PolarsResult<DataFrame>;

    fn get_plan_uuid(&self) -> Uuid {
        panic!("should not get here");
    }

    fn execute_prologue(&self, cache: &mut ExecutionState) -> PolarsResult<()> {
        let plan_uuid = self.get_plan_uuid();
        let uuid = execute_prologue(cache.ctx_id, plan_uuid, cache.active_df_uuid)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        println!("setting to {uuid}");
        cache.set_active_df_uuid(uuid);

        Ok(())
    }

    fn execute_epilogue(&self, cache: &mut ExecutionState) -> PolarsResult<()> {
        let active_df_uuid = execute_epilogue(cache.ctx_id, cache.transform.clone())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        println!("setting to {active_df_uuid}");
        cache.set_active_df_uuid(active_df_uuid);

        Ok(())
    }
}

pub struct Dummy {}
impl Executor for Dummy {
    fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("should not get here");
    }
}

impl Default for Box<dyn Executor> {
    fn default() -> Self {
        Box::new(Dummy {})
    }
}
