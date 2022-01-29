'''
文档：
https://psmatching.readthedocs.io/en/latest/#

github:
https://github.com/rlirey/psmatching


模型使用的方法：glm广义估计
glm_binom = sm.formula.glm(formula = model, data = data, family = sm.families.Binomial())

Match的方式：
同时取K个结果，进行匹配

'''



import psmatching.match as psm
import pytest

# 文件描述: 取营销动作后，有影响的用户历史数据(营销动作前的数据)作为正样本，其他用户历史数据作为负样本，从负样本的用户中，找到和正样本同类的用户
path = r"simMATCH.csv"  # 数据包含3类: 1、OPTUM_LAB_ID 每行唯一标识 2、AGE 到 TOTAL_YRS列 特征(非实验操作的特征) 3、标签(干预项) case 
model = "CASE ~ AGE + TOTAL_YRS"
k = "3"

m = psm.PSMatch(path, model, k)  # 实例化类PSMatch为m，并设置属性值


# Instantiate PSMatch object
# m = PSMatch(path, model, k)

# Calculate propensity scores and prepare data for matching
m.prepare_data()  # 调用实例方法，处理数据

# Perform matching
m.match(caliper = None, replace = False)  # m增加两个变量作为属性: matches: 正样本索引key和负样本索引组成list为values的字典; matched_data:  matches中存在的索引对应的特征等数据

# Evaluate matches via chi-square test
m.evaluate()  

# 通过卡方验证后的数据对，就可以对比这些用户后面经过营销动作后，变化的差异 end 

# 一些数据
# a = m.df # 打上PROPENSITY 潜力分
# m.df['CASE'].value_counts()

# b = m.matched_data   # 匹配之后的数据对
# m.matched_data['CASE'].value_counts()

