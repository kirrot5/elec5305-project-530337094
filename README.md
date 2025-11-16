# elec5305-project-530337094

Name: Kiro Chen
SID: 530337094
Github ID: kirrot5
Github link: https://github.com/kirrot5/elec5305-project-530337094
Video link: https://sydney.instructuremedia.com/embed/ece65e45-0311-4633-af8d-05c12eec9c55  （If the link cannot be opened, please copy and paste it into the address bar.）

This project implements three typical music source separation methods: Nonnegative Matrix Factorization (NMF), Harmonic-Transient Separation (HPSS), and a deep learning-based Demucs model. All methods are tested on the same real pop music track. This repository contains the complete source code, visualization tools, experimental results examples, and video links for demonstrations.

The output results and song examples have also been uploaded to the corresponding folders.

src/                 # NMF / HPSS / Demucs The core code of the three methods
data/                # Input audio
experiments/         # Experimental code
outputs/             # The automatically generated vocal and accompaniment files, along with the resulting image, are displayed after running the program.
main.py              
requirements.txt     

Installation Environment
pip install -r requirements.txt

 Run Code
 nmf                     
 python main.py --method nmf
 
 hpss                    
 python main.py --method hpss
 
 demucs                  
 python main.py --method demucs
