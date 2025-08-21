## Project Development Sprint Summary

This section chronicles the complete technical evolution of the KUMULUS project, detailing the architectural decisions, challenges, and resolutions that transformed it from an initial concept into a production-grade MLOps pipeline.

### Initial Setup & Debugging

The project's initiation phase focused on hardening the development environment and resolving a series of critical dependency conflicts. Key successes included rectifying a `Pydantic` version mismatch, resolving ML library import errors by enforcing compatible versions, and pivoting from a database-centric approach to a more flexible file-based output for local development, which bypassed complex Docker networking issues.

### Data Pipeline Evolution

The data pipeline underwent significant architectural evolution to ensure data quality and mitigate bias. The process began with foundational scripts for data acquisition and preparation. It progressed to a sophisticated data integrity workflow that included programmatic label refinement using `NDVI/NDWI` spectral indices to reduce noise (`06_refine_labels.py`). The most critical advancement was the re-architecture of the data splitting logic (`04_split_dataset.py`), moving from a simple grid-based split to a **randomized spatial split**. This successfully addressed a severe spatial bias discovered in the data, ensuring the training and validation sets are now geographically representative.

### Architectural Bottlenecks & Resolutions

As the pipeline matured, two major performance bottlenecks were identified and resolved with advanced architectural patterns:

1.  **I/O Bottleneck ("Chip-to-Disk"):** The initial strategy of saving pre-processed image "chips" to disk proved to be a brittle and slow I/O bottleneck, causing catastrophic failures when the data distribution changed. This was resolved by implementing a **unified, in-memory pipeline** on Kaggle. This "rasterize-first" approach processes the entire dataset in memory, from rasterization and on-the-fly normalization to dynamic chip generation, eliminating fragile file dependencies and dramatically improving performance.

2.  **CUDA Memory Overflow:** The strategic upgrade to the powerful `EfficientNet-B7` encoder model exceeded the GPU's VRAM capacity. This was resolved by integrating two industry-standard techniques into the PyTorch training loop: **Automatic Mixed-Precision (AMP)**, which halves memory usage by leveraging FP16 precision, and **Gradient Accumulation**, which simulates large, stable batch sizes while maintaining a low memory footprint.

### Final State

The project has successfully culminated in a production-grade, architecturally sound MLOps pipeline. The final system is a unified, in-memory workflow optimized for performance, scalability, and robustness. It is capable of handling large-scale models and is fully integrated with Optuna for automated hyperparameter optimization, providing the solid foundation required for the next phase of advanced, model-centric experimentation.