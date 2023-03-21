# TR_DATA_ReinforcementLearning
A data science and machine learning project proposed by Mines Paris Sophia, control of an underwater robotic arm by reinforcement learning approach

## Data scraping script: scraping_script.py
* A data scraping script that we could run directly after setting up the python environment.
* It will automatically create two checkpoint files (`visited_product`, `visited_product_dict`) in the folder `data`, in the help of which the script can save intermediate scraping results and continue from there the next time we run it, when we encounter some connection problems or the website is down.
* It will extract all the product information presented on this [website](https://www.huffandpuffers.com/collections/disposable-salt-nicotine-devices?sort_by=best-selling), including all the customer reviews under them, and save the data in form of csv table files in the folder `data`.
* Each time we start running this script, we need to make sure that `data` file is empty, and that the parameter for `init_checkpoint(Flag = True)` is True.
* Each time we want to continue the script from the point where we stopped last time, we need to modify the the parameter for `init_checkpoint(Flag = False)` is False.

## Data analysis notebook: data_analysis.ipynb
* A notebook where all the result is already presented, no need to run from the start.
* We've performed data analysis focusing on several aspects, for example, distribution of products' attributes, influence of single feature on the product's popularity, product feature importance study based on the `random forest` algorithm, `unsupervised classification` of review data, `extraction of keywords`, generation of `word clouds` and `sentiment analysis` for the products' aspect terms, and `trend analysis over time` of volume of interaction of products.<p>