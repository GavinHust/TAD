# Optimal dismantling of directed networks


## Repo Contents
- [Code](https://github.com/GavinHust/TAD/tree/main/Code): The source code in the paper.
- [Data](https://github.com/GavinHust/TAD/tree/main/Data): The data in the paper mainly include the original network data, such as Synthetic network and Real network datasets, as well as the network dismantling result information data used to draw the result pictures in the paper.
  - [Synthetic](https://github.com/GavinHust/TAD/tree/main/Data/Synthetic): The synthetic networks used in the paper, where [ER](https://github.com/GavinHust/TAD/tree/main/Data/Synthetic/ER) includes ER networks with average degrees of 3, 6, 9, and 12, [SF](https://github.com/GavinHust/TAD/tree/main/Data/Synthetic/SF) includes scale-free networks with power-law exponents of 2.2, 2.5, 2.8, and 3.2, and [F](https://github.com/GavinHust/TAD/tree/main/Data/Synthetic/F) includes ER networks and SF networks with different initial F values (ranging from 0.1 to 0.9).
  - [Real](https://github.com/GavinHust/TAD/tree/main/Data/Real): The 15 real networks used in the paper
  - [DNdata](https://github.com/GavinHust/TAD/tree/main/Data/DNdata): The network dismantling information of different methods used to draw the result images in the paper, including data in .npy format and .csv format.
  - Otherdata: Other data generated during the experiment.

## Software Dependencies
Users should first install the following software packages in the virtual environment of python3.7. The version of the software, specifically, is:
```
matplotlib==3.5.1
networkx==2.6.3
numpy==1.21.6
powerlaw==1.5
scipy==1.7.3
seaborn==0.13.2
torch==1.13.1+cu116
torch_geometric==2.3.1
```
We also provide the requirement.txt, and users can simply install it through the following command:
```
pip install -r requirements.txt
```

## Instructions to run
1. The generation of the synthetic network for subsequent testing can refer to the following code file, Or of course, you can also use the synthetic network and the real network we have provided for subsequent testing.
  - [get_SF_ER_graph.py](https://github.com/GavinHust/TAD/blob/main/Code/get_SF_ER_graph.py): Generate SF networks with different power-law exponents and ER networks with different average degree values.
  - [get_F_ER_network.py](https://github.com/GavinHust/TAD/blob/main/Code/get_F_ER_network.py): Generate ER networks with different initial values of F.
  - [get_F_SF_network.py](https://github.com/GavinHust/TAD/blob/main/Code/get_F_SF_network.py): Generate SF networks with different initial values of F.

2. Generate the data of GSCC for each step of network dismantling using synthetic networks and real networks. You can modify the hyperparameters in the code [TAD_analysis.py](https://github.com/GavinHust/TAD/blob/main/Code/TAD_analysis.py) to dismantle different networks, including "ER", "SF", "F_ER", "F_SF" and "real".
```
python TAD_analysis.py
```
3. Test the dismantling performance of the ER and SF networks, and draw fig2 in the paper.
```
python draw_fig2.py
```
4. Draw avalanche diagrams of different dismantling methods, and draw fig3 in the paper.
```
python draw_fig3.py
```
5. Test the dismantling performance of ER and SF networks with different initial values of F, and draw fig4 in the paper.
```
python draw_fig4.py
```
6. Test real data, and draw fig5 and fig6 in the paper.
```
python draw_real.py
```

# Reference

Please cite our work if you find our code/paper is useful to your work. 
