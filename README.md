I've written a *lot* of numerical JAX and PyTorch, now used in diverse applications across ML (large protein models, large language models, ...) and science (simulation of black holes, soil moisture, ...). I would particularly highlight:

1. [**Equinox**](https://github.com/patrick-kidger/equinox): elegant neural networks. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/equinox?style=social)](https://github.com/patrick-kidger/equinox)
    
2. [**Diffrax**](https://github.com/patrick-kidger/diffrax): numerical ODE/SDE solvers. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/diffrax?style=social)](https://github.com/patrick-kidger/diffrax)

3. [**jaxtyping**](https://github.com/patrick-kidger/jaxtyping): shape/dtype annotations for arrays. (Also supports PyTorch etc, despite the name!) [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/jaxtyping?style=social)](https://github.com/patrick-kidger/jaxtyping)

> [!TIP]
> <details><summary><i>Click to expand:</i> a full list of other libraries for JAX / Python / publishing / etc.</summary>
> 
> ### JAX
> 
> 4. [**Lineax**](https://github.com/patrick-kidger/lineax): linear/least-squares solvers. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/lineax?style=social)](https://github.com/patrick-kidger/lineax)
> 
> 5. [**Optimistix**](https://github.com/patrick-kidger/optimistix): root finding, least squares, etc. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/optimistix?style=social)](https://github.com/patrick-kidger/optimistix)
> 
> 6. [**sympy2jax**](https://github.com/patrick-kidger/sympy2jax): optimise your symbolic expressions via gradient descent! [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/sympy2jax?style=social)](https://github.com/patrick-kidger/sympy2jax)
> 
> 7. [**Quax**](https://github.com/patrick-kidger/quax): multiple dispatch in JAX! [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/quax?style=social)](https://github.com/patrick-kidger/quax)
> 
> 8. [**ESM2quinox**](https://github.com/patrick-kidger/esm2quinox): ESM2 implemented in JAX. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/esm2quinox?style=social)](https://github.com/patrick-kidger/esm2quinox)
> 
> ### Python
> 
> 9. [**tinyio**](https://github.com/patrick-kidger/tinyio): Ever used asyncio and wished you hadn't? A tiny (~300 lines) event loop for Python. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/tinyio?style=social)](https://github.com/patrick-kidger/tinyio)
> 
> 10. [**patdb**](https://github.com/patrick-kidger/patdb): A fast, pretty, TUI/REPL Python debugger. Includes syntax highlighting, support for re-raised and grouped exceptions, and is robust to asyncio/threading/multiprocessing. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/patdb?style=social)](https://github.com/patrick-kidger/patdb)
> 
> 11. [**Wadler-Lindig**](https://github.com/patrick-kidger/wadler_lindig): A better Python pretty-printer, based upon the theory of Wadler and Lindig. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/wadler_lindig?style=social)](https://github.com/patrick-kidger/wadler_lindig)
> 
> 12. [**action_update_python_project**](https://github.com/patrick-kidger/action_update_python_project/): GitHub CI/CD to automatically deploy Python projects to PyPI and GitHub when a version is bumped.[![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/action_update_python_project?style=social)](https://github.com/patrick-kidger/action_update_python_project)
> 
> ### Typst
> 
> 13. [**typst-pyimage**](https://github.com/patrick-kidger/typst_pyimage): A Typst extension adding support for generating figures using inline Python code. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/typst_pyimage?style=social)](https://github.com/patrick-kidger/typst_pyimage)
>
> 14. [**typst-marimo**](https://github.com/patrick-kidger/typst_marimo): Typst extension, adding support for generating figures and processing values using a companion Marimo notebook. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/typst_marimo?style=social)](https://github.com/patrick-kidger/typst_marimo)
> 
> 15. [**typsy**](https://github.com/patrick-kidger/typsy): Classes/structs, pattern matching, safe counters... and more! Your one-stop library for programming tools not already in core Typst. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/typsy?style=social)](https://github.com/patrick-kidger/typsy)
> 
> ### MkDocs
> 
> 16. [**MkPosters**](https://github.com/patrick-kidger/mkposters): Write academic posters in Markdown, style them with CSS, save them to PDF. No wrestling with LaTeX. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/mkposters?style=social)](https://github.com/patrick-kidger/mkposters)
> 
> 17. [**mkdocs_ipynb**](https://github.com/patrick-kidger/mkdocs_ipynb/): Use `*.ipynb` files (Jupyter notebooks) when building documentation with MkDocs. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/mkdocs_ipynb?style=social)](https://github.com/patrick-kidger/mkdocs_ipynb)
> 
> 18. [**mkdocs_include_exclude_files**](https://github.com/patrick-kidger/mkdocs_include_exclude_files/): Include or exclude specific files when building documentation with MkDocs. [![GitHub Repo stars](https://img.shields.io/github/stars/patrick-kidger/mkdocs_include_exclude_files?style=social)](https://github.com/patrick-kidger/mkdocs_include_exclude_files)
> 
> ### Julia
> 
> 19. [FromFile.jl](https://github.com/Roger-luo/FromFile.jl): An improved import+include system for Julia. Makes your files self-contained and easier to understand. [![GitHub Repo stars](https://img.shields.io/github/stars/Roger-luo/FromFile.jl?style=social)](https://github.com/Roger-luo/FromFile.jl)
> 
> </details>


### Me:

I am currently a tech lead on ML for protein engineering (lead optimization) at [Cradle Bio](https://cradle.bio), and founded much of the open-source scientific JAX ecosystem. I also hold an honorary lectureship at Imperial College London. I previously worked at Google X, and received my PhD from Oxford on neural differential equations.

My current interests include pretty much anything related to scientific machine learning and scientific computing! I've now worked across diverse parts of the field, from modern deep learning (protein language models) to classical methods (numerics), to everything in between (neural differential equations).

I am also known for having strong opinions on the importance of good software development! :)

**Other links:**

- Bluesky: [![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&label=Follow%20me%20on&color=0285FF)](https://bsky.app/profile/PatrickKidger.bsky.social)
- Twitter: [![Twitter Follow](https://img.shields.io/twitter/follow/PatrickKidger?style=social)](https://twitter.com/PatrickKidger)
- Google scholar: [here](https://scholar.google.co.uk/citations?user=5cCLsNQAAAAJ)
- Personal website: [kidger.site](https://kidger.site)
- Neural ODE/SDE textbook: [arXiv/2202.02435](https://arxiv.org/abs/2202.02435)
