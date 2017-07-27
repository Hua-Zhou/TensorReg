---
layout: default
title: TensorReg
---

## TensorReg Toolbox for Matlab

TensorReg toolbox is a collection of Matlab functions for tensor regressions.

The toolbox is developed by [Hua Zhou](http://hua-zhou.github.io).  

### Compatibility

The code is tested on Matlab R2017a, but should work on other versions of Matlab with no or little changes. Current version works on these platforms: Windows 64-bit, Linux 64-bit, and Mac (Intel 64-bit). Type `computer` in Matlab's command window to determine the platform.

### Dependency

TensorReg toolbox uses the Tensor Toolbox. Please install the [Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html) before installing the TensorReg toolbox.

### Installation (Matlab version >= 2014b)

Download the Matlab toolbox installation file [TensorReg.mltbx](./TensorReg.mltbx). Double click the downloaded file and you should be good to go. If it does not work for some reasons, follow the below instructions for Matlab version < 2014b.


### Installation (Matlab version < 2014b)

1. Download `ZIP File` file using the links on the left.  2. Extract the zip file.  
```
unzip Hua-Zhou-TensorReg-xxxxxxx.zip
```
3. Rename the folder from *Hua-Zhou-SparseReg-xxxxxxx* to *SparseReg*.  
```
mv Hua-Zhou-TensorReg-xxxxxxx TensorReg
```
4. Add the *TensorReg* folder to Matlab search path. Start Matlab, cd to the *TensorReg* directory, and execute the following commands  
`addpath(pwd)	 %<-- Add the toolbox to the Matlab path`  
`save path	 %<-- Save for future Matlab sessions`
5. Go through following tutorials for the usage. For help of individual functions, type `?` followed by the function name in Matlab.

### Tutorial

* [Resize arrays](./html/demo_resize.html)
* [Kruskal (CP) regression](./html/demo_kruskal.html)
* [Tucker regression](./html/demo_tucker.html)   
* [Regularized matrix regression](./html/demo_matrixreg.html)


### How to cite

If you use this toolbox, please cite the software itself along with at least one publication or preprint.

* Software reference:  
H Zhou. Matlab TensorReg Toolbox Version 1.0, Available online, March 2017.  
* Default article to cite for Kruskal (CP) tensor regression:    
H Zhou, L Li, and H Zhu (2013) Tensor regression with applications in neuroimaging data analysis, [_Journal of American Statistical Association_](http://www.tandfonline.com/doi/abs/10.1080/01621459.2013.776499#.UeW24mTXjbw), 108(502):540-552.  
* Default article to cite for Tucker tensor regression:    
X Li, H Zhou, and L Li (2013) Tucker tensor regression and neuroimaging analysis, \[[arXiv:1304.5637](http://arxiv.org/abs/1304.5637)\].    
* Default article to cite for regularized matrix regression:    
H Zhou and L Li. (2014) Regularized matrix regression, [_Journal of Royal Statistical Society Series B_](http://onlinelibrary.wiley.com/doi/10.1111/rssb.12031/abstract), 76(2):463-483.  

### Contacts

Hua Zhou <huazhou@ucla.edu>
