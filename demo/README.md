# Demo: music-outliers

An [interactive visualisation](https://mpanteli.github.io/music-outliers/demo/outliers.html) of outlier music recordings ([source code](https://github.com/mpanteli/music-outliers/blob/master/scripts/interactive_plot.py)).

## Background

Recordings from a world music corpus were compared and outlier detection methods assessed the distinctiveness of their musical characteristics. The dataset included 8200 recordings from 137 countries sampled from the Smithsonian Folkways Recordings and the World and Traditional Music Collection of the British Library Sound Archive. Several music features were taken into account for comparison: rhythmic, harmonic, melodic, timbral and a combination of all these ('All'). Mahalanobis distances assessed how distinct the feature vector of each recording is compared to the set of recordings. Recordings lying far from the mean of the distribution (large Mahalanobis distances based on a threshold) are considered 'outliers'.

## Figure explained

A total of 8200 recordings from 137 countries are drawn on this map. 

Each recording is represented as a point **drawn at random** within the boundaries of the country it originates from. 

**The colour** of each data point denotes its distinctiveness to the corpus: 
- light grey points, non-outliers, lie close to the mean of the distribution
- dark red points, outliers, lie far from the mean of the distribution

**Hover your mouse** over each data point to get additional information on: 
- the country, e.g., *United States of America*,
- culture, e.g., *Anglo-American*,
- language, e.g., *English*,
- music genre, e.g., *Historical Song*,
- the collection it came from, e.g., *Smithsonian Folkways*, 
- whether this recording was detected as an outlier, e.g., *non-outlier*, 
- and its corresponding Mahalanobis Distance (MD), e.g., *MD=306*. 

![alt tag](https://raw.githubusercontent.com/mpanteli/music-outliers/master/demo/hover-mouse.png)

**Click** on each point to listen to the audio (redirects to the Smithsonian Folkways and British Library websites). 

**Choose tab** *Rhythm, Melody, Timbre, Harmony* to see the outcome of music outliers with respect to rhythmic, melodic, timbral, and harmonic features. Default tab *All* shows music outliers when all features are combined. 

![alt tag](https://raw.githubusercontent.com/mpanteli/music-outliers/master/demo/tabs.png)

**Explore interactivity** such as *Pan, Box zoom, Save* with the tools on the right. 

![alt tag](https://raw.githubusercontent.com/mpanteli/music-outliers/master/demo/interactivity.png)
