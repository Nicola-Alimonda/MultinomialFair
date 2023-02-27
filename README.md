# MultinomialFair
Repository of experiments for a Fair Top-k Ranking problem.

> ### Authors & contributors:
> Nicola Alimonda, Alessandro Castelnovo, Riccardo Crupi

To know more about this research work, please refer to:

- Zehlike, Meike, et al. "Fa* ir: A fair top-k ranking algorithm." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
- Alimonda, Nicola, et al. "Preserving Utility in Fair Top-k Ranking with Intersectional Bias." Working in progress. 2023.


### Using the fair ranking algorithm
Clone repo and install packages:
```
git clone https://github.com/Nicola-Alimonda/MultinomialFair
cd multinomialFair
pip install -r requirements.txt
```

Python version: `3.8.10`

### Run FA*IR

To reprocuce the experiments run:
```
python multinomialFair\test.py
```
In the folder `Minimum Target_tables` there are the minimum targets to ensure correctness for the binomial multinomial distribution.

In `Plots` folder there are the plots of the distributions of the classes by Score and the scatterplots of the score compared to the position of the pre and post optimization rank.

In the `Ranking folder` we find the dataframe post_opt_dataset which evaluates the difference in utility*, the old k position against the new k position, per each individual. Moreover, there is the same dataframe that shows only the utilities but aggregated by class {column: Group}.

<img src="https://user-images.githubusercontent.com/92302358/220327881-52c5acc0-0a92-418d-a3e8-d26b921c8839.png" width="600" height="600">

Notice that individual utility differences are calculated through (1) a difference between the same individual (pre and post rank {column: Utility_Loss_individual_perc },
(2) and calculated through a difference between the individual who occupies the post opt position and the individual who occupied that position pre-opt {column:Utility_Loss_position_perc}.


### Greedy Wise Score
With this work we propose a greedy extension of the previous algorithm.

Inside the Greedy-Wise-Score folder you find packages used to run Greedy_Wise_Score.py and Greedy_Wise_Utility.py. Inside the Greedy multinomial Fair-Copy1.ipynb notebook, on the other hand, there are all the tests carried out and the plots generated.

```
python Greedy-Wise-Score\Greedy_Wise_Score.py 
python Greedy-Wise-Score\Greedy_Wise_Utility.py
```

Under Analisi di sensitività congiunta per alpha e L, inside the notebook, is possible to set initialization parameter such as k (lenght of the ranking), p (given proportions of each protected group), and attribute for the number of protected groups.

Dataframe df_res returns all possible combinations of ranking with given parameter and the obtained kendal-tau metric for each ranking.
In dataframe exposureDf is possible to see average exposure for each group G. 
