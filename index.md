<img src="images/banner.png" alt="hi" class="inline"/>

# What's at stake
A civil war is defined as a conflict between different groups of a single state or country. Such wars can have serious repercussions like resources´ consumption, crowd movement, or destruction of infrastructures. Besides, most of these conflicts imply international intervention -they represented two thirds of the 138 intrastate conflicts between the end of World War II and 2000 [[1]](#1). Thereby, understanding the dynamics leading to these rare events happens to be crucial. It would allow to anticipate and even to prevent civil wars in the years to come.  

# What has already been done to predict civil war onset 
Recent advancements in data science have catalyzed the development of predictive models for various problems, including political conflicts such as civil wars [[2]](#2) [[3]](#3) [[4]](#4). In 2015, Muchlinski _et al._ published a paper _Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data_. They used Random Forests to try predicting rare events in conflict data more accurately. 
In this project, we first aim at benchmarking these results against other common machine learning algorithms, namely neural networks and gradient boosted trees. As the geographical area was an important feature in the abovementioned study ("Western Europe and US Dummy" in Fig. 4 [[5]](#5)), we will examine whether predictions differ when training separate models for each continent or subregion. Finally, we will explore whether civil war events can be forecast as time series instead of simple regression points, e.g., using autoregressive models.

# Stretching the limits of previous analysis 
## Comparing machine learning algorithms for civil war onset classification.
Using the same features than Muchlinski _et al._ [[5]](#5) minus the features related to geographical area separation (e.g.,  "Western Europe and US Dummy"). Indeed, we will investigate the effect of data aggregation by continent/subregion in a subsequent analysis (cf. **4.2**). To ensure a meaningful comparison, the same training and testing sets will be used to assess each tested algorithm.

## Effect of spatial data separation
Following the importance of the "Western Europe and US Dummy" variable in Muchlinski _et al._ [[5]](#5), we decided to aggregate the data by subregions, and to see whether predictive accuracy differs when fitting the models separately on each group. We then analyzed feature importance to see whether different variables can explain better civil war events in different geographical areas.

 ## Predicting civil war onset as time series forecasting
In this section, we considered that each subregion is assigned to a time series of civil war/peace events. This part of the project aimed at seeing whether civil war events could be forecast given past data using simple time forecasting models (e.g., autogressors (AR) or ARIMA). 



## References
<a id="1">[1]</a> Civil Wars and Foreign Powers: Outside Intervention in Intrastate Conflict. By Patrick M. Regan. Ann Arbor: University of Michigan Press, 2000.

<a id="2">[2]</a> Fearon, J. D., & Laitin, D. D. (2003). Ethnicity, insurgency, and civil war. _American political science review_, 75-90.

<a id="3">[3]</a> Collier, P., & Hoeffler, A. (2004). Greed and grievance in civil war. _Oxford economic papers_, 56(4), 563-595.

<a id="4">[4]</a> Hegre, H., & Sambanis, N. (2006). Sensitivity analysis of empirical results on civil war onset. _Journal of conflict resolution_, 50(4), 508-535.

<a id="5">[5]</a> Muchlinski, D., Siroky, D., He, J., & Kocher, M. (2016). Comparing random forest with logistic regression for predicting class-imbalanced civil war onset data. _Political Analysis_, 87-103.

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/Neclow/ada-2020-project-milestone-p3-p3_navaja/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Neclow/ada-2020-project-milestone-p3-p3_navaja/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
