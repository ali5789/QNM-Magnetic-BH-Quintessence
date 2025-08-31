# QNM-Magnetic-BH-Quintessence
Code, data, and LaTeX source for the paper  "Quasinormal Modes of Magnetically Charged Black Holes in Quintessence" by Ali Hasnain.

This repository contains the source code, data, and LaTeX source for the paper:

The work presents a numerical study of quasinormal mode (QNM) spectra for a static, magnetically charged black hole solution within nonlinear electrodynamics, embedded in a quintessence field. The code here allows full reproducibility of the results presented in the paper.

⚠️ Note on Accuracy  
This repository uses a simplified 1st-order WKB approximation for reproducibility.  
The exact values in the paper were obtained using higher-order methods, so small numerical differences are expected.  
Numerical stability note: For some combinations with large quintessence or charge, the effective potential may not form a clean barrier outside the horizon. In such cases the 1st-order WKB routine prints a clear message and skips that run. Try slightly smaller --quintessence or --charge, or higher multipole --multipole 3


 # Repository Structure
QNM-Magnetic-BH-Quintessence/
├── paper/
│   └── Quasinormal_Modes_of_Magnetically_Charged_BH_in_Quintessence.pdf
├── src/
│   ├── calculate_qnms.py
│   ├── geometry.py
│   ├── potentials.py
│   ├── wkb.py
│   └── requirements.txt
├── data/
│   └── schwarzschild.dat
├── README.md
└── LICENSE

# Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ali5789/QNM-Magnetic-BH-Quintessence.git
cd QNM-Magnetic-BH-Quintessence/src
pip install -r requirements.txt



