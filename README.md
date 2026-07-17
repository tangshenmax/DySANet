
# DySA-Net

Official repository for the paper:

**Medical Image Segmentation via Dynamic Feature Synergy, Deformable Semantic Alignment and Context Perception Modulation**

## Status

The manuscript is currently under review.

To ensure reproducibility while respecting the publication process, the complete implementation, pretrained model weights, configuration files, and detailed documentation will be released **immediately upon acceptance** of the paper.

Thank you for your interest in our work.

## Overall Architecture
![Overall](assets/images/DySANet.png)

## SegVis
![SegVis](assets/images/Skin_Seg_Vis.png)

## Dataset
```bibtex
- datasets
    - train
      - images
      - masks
    - val
      - images
      - masks
    - test
      - images
      - masks
```bibtex

[BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
```bibtex
@article{ALDHABYANI2020104863,
	title = {Dataset of breast ultrasound images},
	journal = {Data in Brief},
	volume = {28},
	pages = {104863},
	year = {2020},
	issn = {2352-3409},
	author = {Walid Al-Dhabyani and Mohammed Gomaa and Hussien Khaled and Aly Fahmy},
}
```

[BUSBRA](https://zenodo.org/records/8231412)
```bibtex
@article{gomez2024bus,
	title={BUS-BRA: a breast ultrasound dataset for assessing computer-aided diagnosis systems},
	author={G{\'o}mez-Flores, Wilfrido and Gregorio-Calas, Maria Julia and Coelho de Albuquerque Pereira, Wagner},
	journal={Medical Physics},
	volume={51},
	number={4},
	pages={3110--3123},
	year={2024},
	publisher={Wiley Online Library}
}
```

[STU](https://drive.google.com/file/d/1k3OvEnYZaPWrng74aP4hAhgPXNHjpPj3/view?usp=drive_link)
```bibtex
@article{zhuang2019rdau,
	title={An RDAU-NET model for lesion segmentation in breast ultrasound images},
	author={Zhuang, Zhemin and Li, Nan and Joseph Raj, Alex Noel and Mahesh, Vijayalakshmi GV and Qiu, Shunmin},
	journal={PloS one},
	volume={14},
	number={8},
	pages={e0221535},
	year={2019},
	publisher={Public Library of Science San Francisco, CA USA}
}
```

[Kvasir](https://datasets.simula.no/kvasir-seg/)
```bibtex
@inproceedings{jha2019kvasir,
	title={Kvasir-seg: A segmented polyp dataset},
	author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and De Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
	booktitle={International conference on multimedia modeling},
	pages={451--462},
	year={2019},
	organization={Springer}
}
```

[ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
```bibtex
@article{bernal2015wm,
	title={WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians},
	author={Bernal, Jorge and S{\'a}nchez, F Javier and Fern{\'a}ndez-Esparrach, Gloria and Gil, Debora and Rodr{\'\i}guez, Cristina and Vilari{\~n}o, Fernando},
	journal={Computerized medical imaging and graphics},
	volume={43},
	pages={99--111},
	year={2015},
	publisher={Elsevier}
}
```

[ColonDB](https://www.kaggle.com/datasets/longvil/cvc-colondb)
```bibtex
@article{tajbakhsh2015automated,
	title={Automated polyp detection in colonoscopy videos using shape and context information},
	author={Tajbakhsh, Nima and Gurudu, Suryakanth R and Liang, Jianming},
	journal={IEEE transactions on medical imaging},
	volume={35},
	number={2},
	pages={630--644},
	year={2015},
	publisher={IEEE}
}
```


[ISIC2016](https://challenge.isic-archive.com/data/#2016)
```bibtex
@article{gutman2016skin,
	title={Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (ISBI) 2016, hosted by the international skin imaging collaboration (ISIC)},
	author={Gutman, David and Codella, Noel CF and Celebi, Emre and Helba, Brian and Marchetti, Michael and Mishra, Nabin and Halpern, Allan},
	journal={arXiv preprint arXiv:1605.01397},
	year={2016}
}
```
[ISIC2018](https://challenge.isic-archive.com/data/#2018)
```bibtex
@article{codella2019skin,
	title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic)},
	author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and Celebi, M Emre and Dusza, Stephen and Gutman, David and Helba, Brian and Kalloo, Aadi and Liopyris, Konstantinos and Marchetti, Michael and others},
	journal={arXiv preprint arXiv:1902.03368},
	year={2019}
}
```

[PH2](https://www.kaggle.com/datasets/athina123/ph2dataset)
```bibtex
@inproceedings{mendoncca2013ph,
	title={PH 2-A dermoscopic image database for research and benchmarking},
	author={Mendon{\c{c}}a, Teresa and Ferreira, Pedro M and Marques, Jorge S and Marcal, Andr{\'e} RS and Rozeira, Jorge},
	booktitle={2013 35th annual international conference of the IEEE engineering in medicine and biology society (EMBC)},
	pages={5437--5440},
	year={2013},
	organization={IEEE}
}
```
