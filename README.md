# Titanic passengers survival analysis Kaggle competition

In this notebook we will conduct data analysis and predictive modelling of Titanic passengers survival dataset for Kaggle [Titanic dataset](https://www.kaggle.com/c/titanic).

Titanic passengers survival dataset is one of the most canonical data analysis and machine learning datasets. Let's see how this data looks like, investigate main trends in it and try to predict survival chances of passengers based on numerous data attributes.

To see the analysis open above Jupyter notebook file or click [here](https://github.com/msznajder/titanic_survival_analysis_kaggle/blob/master/titanic_passengers_survival_analysis_kaggle_competition.ipynb).

## Framing the problem

Titanic sinking was one of the biggest shipwreck tragedies in history. It killed 1502 out of 2224 passengers. This tragedy led to better safety regulations for ships. The main reason for such a great death toll was not enough number of lifeboats to fit all passengers. This can lead to the conclusion that some groups of people were more likely to survive than others - woman, children and upper-class passeners for example.

### The goal

The goal of this analysis is two fold. Firstly, we want to analyse the data set, explore it answering related questions using data visualization and statistical methods. Secondly our goal is to predict, by building machine learning model, which passengers were likely to survive Titanic disaster and which not.

#### Analysis

There are some questions we would like to answer with Titanic dataset analysis.

* What is Titanic passengers demographic structure analyzed in terms of attributes?

* What is the overall Titanic passengers survival ratio?

* What is the survival ratio for different demographic passengers groups? Which groups have biggest chances for survival and which smallest? Is the difference between survived group statistically significant?

#### Modeling

As for second goal, since we are given multiple attributes with labeled survival target values this is clearly an supervised learning problem. Moreover, we try to predict discrete categorical value: 0 or 1 (representing not-survived and survived passenger). This tells us that we deal here with binary categorization problem.

For each `PassengerId` in the provided unlabeled test set, we want to predict a whether given passenger survived or not.

As a measure of created machine learning models performance we choose the percentage of passengers we correctly predict, that is accuracy level.

As a prediction final product we will prepare and submit to Kaggle a csv file with predicted survival for exactly 418 entries in the test dataset. The file should have exactly 2 columns:
* PassengerId (sorted in any order)
* Survived (contains your binary predictions: 1 for survived, 0 for deceased)

The example submission dataset should look like this:
```
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
```

## Summary

In this analysis we worked with Titanic survival dataset for 891 passengers data out of all 2224 Titanic passengers.

We first answered what was Titanic passengers demographic structure analyzed in terms of attributes. We saw that vast majority (55.11%) of passengers travelled in lowest third class, almost 25% of passengers travelled in first class and 20.65% traveled in second class. Majority of all passengers were males: 64.76% and only 35.24% of females. 10.78% of passengers were children in age of 0-14. A little over 28% of passengers were both between 14-24 and 24-34 years old. Almost 17% of all passengers were 34-44 years old. Only 16% of all passengers were above 44 years old. That tells us that Titanic passengers population was quite young.

Vast majority of passengers travelled without any siblings or spouse - 68.24% of them. 23.46% of passengers travelled with one child or spouse. Less than 9% of all passengers traveled with more than one sibling or spouse. The situation is very similar in case of passengers travelling with parents or children. Most of them, 76.09%, travelled alone and 13.24% travelled with just one parent or children. 8.89% of passengers travelled with two parents or children. In terms of ticket fare over the half, 57.08%, of passengers paid the lowest fare ranging from 0 to 20. 22.83% passengers paid between 20 and 40 fare. We saw that the rest of the ticket prices, 20.11%, varies very much ranging from 40 to 600. Finally we can see that vast majority of passengers, 72.44%, embarked in Southampton, 18.9% embarked in Cherbourg and only 8.66% of passengers embarked Titanic in Queenstown.

In this part of analysis by accident we found error in the Titanic dataset. We found out that the age of the person appearing in the dataset as the oldest (80) is the age of actual death many years after person disaster survival. This means that dataset age information for this passenger, by error, contains age value of the post Titanic death and not the day of disaster age as it does for other survived passengers age attribute. This inconsistency makes the data invalid and confusing when it comes to factual or historical value.

We answered what is the overall Titanic passengers survival ratio. We found out that from Titanic 891 passengers only 342 survived and 549 died. The Titanic survival ratio is approximately 0.3838, meaning that only 38.38% of all passengers survived the disaster.

Finally we turned to the analysis of survival ratio for different demographic passengers groups. As for passengers classes survival ratio, the survival ratio for the first class passengers was 0.63. Second class had a bit lower survival ratio of 0.47. But comparing that to the third class passengers survival ratio of 0.24 is shocking. All three passengers classes effects are significant with p < .05. Looking at gender based survival differences we also see clear relationship in terms of survival ratio values. Clearly women were more likely to survive than men with 0.74 survival ratio for women and 0.18 for men. Both gender effects are significant with p < .05. As for age groups survival ratio passengers in the age group of 0-14 have much higher survival ratio of 0.58 than the rest of passengers. Other passengers age groups oscillate around the overall survival ratio of 0.38. All ages groups effects besides (54, 64] group are significant with p < .05.

As for passengers travelling with different number of siblings or spouse, passengers travelling without any had survival ratio of 0.35. Passengers travelling with one and two sibling or spouse had higher survival chances of 0.54 and 0.46. Only passengers travelling without any and with one sibling/spouse effects are significant with p < .05. Considering groups of passengers depending on the number of parents and children they travelled with, again passengers travelling alone had survival ratio of 0.34. We can see that passengers travelling with one, two or three children or parents had bigger chances of survival, 0.55, 0.50 and 0.60 accordingly, but again let's remember that these groups had much less passengers. Only passengers travelling without any and with one or two children/parents effects are significant with p < .05.

When we look at ticket fare attribute survival ratio values we can see clearly almost linear relation between the price of the ticket and the survival ratio. It looks like the more passenger paid for the ticket the bigger were chances of survival. The survival ratio for passengers who bought the tickets for 0-20 was 0.28, for passengers with tickers between 20-40 the survival ratio was 0.43. And for further groups: 40-60 0.57, 60-80 0.520833, 80-100 0.86, 100-300 0.72. For tickets with price between 300-600 the survival ratio was 1.00 meaning that all of these passengers survived. All passengers tickets fares groups effects are significant with p < .05. Finally we looked at the Embarked attribute passengers groups. Passengers who embarked in Southampton had the lowest survival ratio out of all three ports of embarkation (0.34). Similarly passengers who embarked in Queenstown had survival ratio equal to 0.39. However those who embarked in Cherbourg had survival ratio of 0.55. We confirmed it with data that the third class passengers embarked mostly in Southampton and Queenstown and that could be the reason for such small, as compared to Cherbourg port, survival ratio for passengers embarking in Southampton. All passengers embarkation ports effects are significant with p < .05.

Finally we modeled our data using various machine learning algorithms. As a result of performed tests and hyperparameters tuning Random Forest model was proven to give best cros-validation accuracy score for survival predictions of 0.82716. After evaluating our final model on test dataset our model reached 0.74162 accuracy score. This is quite good result but also leaves quite a lot of space for improvements.
