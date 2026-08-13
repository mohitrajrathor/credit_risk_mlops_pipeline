# Credit Risk Pipeline - Test Execution Report

**Status**: SUCCESS

### Summary Metrics
- **Total Tests**: 12
- **Passed**: 12
- **Failed**: 0
- **Skipped**: 0
- **Errors**: 0
- **Execution Time**: 4.15 seconds

### Test Case Statuses
| Test Case | Status | Duration (s) |
| :--- | :--- | :--- |
| `tests/test_pipeline.py::test_config_loader` | passed | 0.00 |
| `tests/test_pipeline.py::test_logger` | passed | 0.00 |
| `tests/test_pipeline.py::test_ingest_data` | passed | 0.01 |
| `tests/test_pipeline.py::test_validate_data` | passed | 0.01 |
| `tests/test_pipeline.py::test_transform_data` | passed | 0.02 |
| `tests/test_pipeline.py::test_feature_engineering` | passed | 0.02 |
| `tests/test_pipeline.py::test_build_models` | passed | 0.00 |
| `tests/test_pipeline.py::test_train_models` | passed | 0.37 |
| `tests/test_pipeline.py::test_evaluate_models` | passed | 0.16 |
| `tests/test_pipeline.py::test_register_best_model` | passed | 0.00 |
| `tests/test_pipeline.py::test_shap_explainer` | passed | 0.42 |
| `tests/test_pipeline.py::test_main_pipeline_e2e` | passed | 1.03 |