# elec5305-project-530337094

Name: Kiro Chen
SID: 530337094
Github ID: kirrot5
Github link: https://github.com/kirrot5/elec5305-project-530337094

Project Overview
This project aims to develop a system that can automatically separate mixed music into vocals and accompaniment. This task has significant application value in scenarios such as karaoke, music production, and audio editing. The project will initially utilize short-time Fourier transform (STFT) and spectral masking methods to achieve baseline separation, and further explore deep learning methods based on U-Net and Demucs to improve the separation quality. By comparing traditional and deep learning methods and conducting experiments and evaluations on the MUSDB18 dataset, the project will develop a functional prototype for automatic music separation and demonstrate its potential value in accompaniment generation and intelligent audio applications.

Background
Music signal separation is an important research direction in audio signal processing, aiming to extract different sound sources from mixed audio, such as human voices and accompaniment. This issue has practical application value in karaoke, music production, and information retrieval. Early methods mainly relied on traditional signal processing and matrix decomposition, such as non-negative matrix factorization (Lee & Seung, 1999; Virtanen, 2007) and independent component analysis. However, their performance was limited in complex music scenarios.
In recent years, deep learning methods have significantly improved the separation performance. The U-Net network achieved excellent results in human voice separation (Jansson et al., 2017), while the Demucs model performed outstandingly in international evaluations through time-domain modeling (Defossez et al., 2019). Nevertheless, existing models still face challenges such as insufficient generalization ability, high computational cost, and artifact problems.
