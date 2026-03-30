# TADF Emitters: Computational Screening and Design Guidelines

This repository contains the manuscripts and supporting materials for two related research articles on thermally activated delayed fluorescence (TADF) emitters, based on high-throughput computational screening of 747 experimentally known molecules.

## Overview

| Article | Focus | Status | Key Contribution |
|---------|-------|--------|------------------|
| **Article 1** | Method Validation | ✅ Published | Validates xTB-based workflow for TADF screening (>99% cost reduction) |
| **Article 2** | Design Guidelines | ✅ Published  | Extracts quantitative design rules and identifies 127 high-performance candidates |

## Articles

### Article 1: xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark

**Status:** Published  
**Journal:** Journal of Chemical Information and Modeling (2026)  
**DOI:** [10.1021/acs.jcim.5c02978](https://doi.org/10.1021/acs.jcim.5c02978)  
**arXiv:** [2511.00922](https://arxiv.org/abs/2511.00922)

This article validates semi-empirical sTDA-xTB and sTD-DFT-xTB methods for high-throughput screening of TADF emitters using 747 experimentally characterized molecules—the largest benchmark to date. The framework achieves >99% computational cost reduction versus TD-DFT while maintaining strong internal consistency and reasonable agreement with experimental data.

**Folder:** `Archive1_xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark/`

### Article 2: Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules

**Status:** ✅ Published  
**Journal:** Journal of Chemical Information and Modeling (2026)  
**DOI:** [10.1021/acs.jcim.5c03068](https://doi.org/10.1021/acs.jcim.5c03068)  
**arXiv:** [2511.11606](https://arxiv.org/abs/2511.11606)

This article leverages the validated computational workflow from Article 1 to extract quantitative design guidelines for TADF emitters. Through systematic analysis of molecular architecture, geometry, and electronic structure, it identifies 127 high-performance candidates and establishes structure-property relationships to guide future TADF development.

**Folder:** `Article2_Data-Driven Design Guidelines for TADF-Emitters from a High-Throughput Screening of 747 Molecules/`

## Relationship Between Articles

These two articles form a cohesive research program:

1. **Article 1** establishes and validates the computational methodology (xTB-based high-throughput screening)
2. **Article 2** applies this validated methodology to extract design rules and identify promising candidates

Both articles analyze the same dataset of 747 TADF molecules and share computational infrastructure.

## How to Use This Repository

**If you want to:**
- **Validate computational methods** for TADF screening → See Article 1
- **Apply design guidelines** for new TADF molecules → See Article 2
- **Access raw computational data** → Visit the [GitHub repository](https://github.com/TchapetNjafa/Data_Articles_TADF)
- **Reproduce calculations** → See the computational pipeline in the GitHub repository
- **Cite this work** → Use the BibTeX entries below for the relevant article(s)

## Related GitHub Repository

The complete computational data, scripts, and results supporting both articles are available in the public GitHub repository:

**Repository:** [smiEmpirical-TADF](https://github.com/TchapetNjafa/Data_Articles_TADF)  
**Data Location:** `Public_Results/Result_article1_TADF_xTB/`  
**DOI:** [10.5281/zenodo.17436069](https://doi.org/10.5281/zenodo.17436069)

The repository includes:
- Raw computational data for all 747 molecules (gas phase and toluene solvent)
- Complete computational pipeline scripts (xTB, CREST, sTDA, Multiwfn)
- Machine learning reproducibility code
- High-level theory validation calculations (OT-LC-PBE)

## Authors

- Jean-Pierre Tchapet Njafa (jean-pierre.tchapet@facsciences-uy1.cm)
- Elvira Vanelle Kameni Tcheuffa
- Aissatou Maghame Foumkpou
- Serge Guy Nana Engo

Department of Physics, Faculty of Science, University of Yaounde I, Cameroon

## Frequently Asked Questions

**Q: Which article should I cite?**  
A: If you use the computational methodology or validation results, cite Article 1. If you use the design guidelines or candidate molecules, cite Article 2. For comprehensive work, cite both.

**Q: Where can I find the raw computational data?**  
A: All data is in the [GitHub repository](https://github.com/TchapetNjafa/Data_Articles_TADF) under `Public_Results/Result_article1_TADF_xTB/`.

**Q: Can I use the computational workflow for my own molecules?**  
A: Yes! The complete pipeline is available in the GitHub repository with documentation.

**Q: What software do I need to reproduce the calculations?**  
A: See the [repository README](https://github.com/TchapetNjafa/Data_Articles_TADF/blob/main/Public_Results/Result_article1_TADF_xTB/README.md) for complete software requirements and installation instructions.

**Q: Are the 127 high-performance candidates available?**  
A: Yes, they are identified in Article 2 and the data is available in the GitHub repository.

## Citation

If you use these manuscripts or the associated data in your research, please cite:

**Article 1:**
```bibtex
@article{tchapet2026validation,
  title={xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c02978}
}
```

**Article 2:**
```bibtex
@article{tchapet2026design,
  title={Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c03068}
}
```

## License

MIT License - See individual article folders for details.

---

*Last updated: March 2026*
