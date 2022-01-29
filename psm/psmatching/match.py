import pandas as pd
import numpy as np
from psmatching.utilities import *


####################################################
###################  Base Class  ###################
####################################################


class PSMatch(object):
    '''
    Parameters
    ----------
    file : string
        The file path of the data; assumed to be in .csv format.
    model : string
        The model specification for calculating propensity scores; in the format Y ~ X1 + X2 + ... + Xn
    k : string
        The number of controls to be matched to each treated case.
    '''

    def __init__(self, path, model, k):
        self.path = path  # 数据文件路径
        self.model = model  # 模型文件需要处理的列名
        self.k = int(k)  # k=3, 表示一个正样本对应几个负样本


    def prepare_data(self, **kwargs):
        '''
        Prepares the data for matching.

        Parameters
        ----------
        path : string
            The file path of the data to be analyzed. Assumed to be in .csv format.

        Returns
        -------
        A Pandas DataFrame containing raw data plus a column of propensity scores.
        '''
        # Read in the data specified by the file path
        df = pd.read_csv(self.path)  # 读取数据文件
        df = df.set_index("OPTUM_LAB_ID")  # 设置用户id列为行索引
        # Obtain propensity scores and add them to the data
        print("\nCalculating propensity scores ...", end = " ")
        propensity_scores = get_propensity_scores(model = self.model, data = df, verbose = False)  # 获取倾向性得分
        print("DONE!")
        print("Preparing data ...", end = " ")
        df["PROPENSITY"] = propensity_scores  
        # Assign the df attribute to the Match object
        self.df = df  # 存储结果的变量作为属性赋给实例
        print("DONE!")


    def match(self, caliper = None, replace = False, **kwargs):
        '''
        Performs propensity score matching.

        Parameters
        ----------
        df : Pandas DataFrame
            the attribute returned by the prepare_data() function

        Returns
        -------
        matches : Pandas DataFrame
            the Match object attribute describing which control IDs are matched
            to a particular treatment case.
        matched_data: Pandas DataFrame
            the Match object attribute containing the raw data for only treatment
            cases and their matched controls.
        '''
        # Assert that the Match object has a df attribute
        if not hasattr(self, 'df'):  # 判断对象是否有df属性
            raise AttributeError("%s does not have a 'df' attribute." % (self))

        # Assign treatment group membership
        groups = self.df.CASE  # 取出标签列
        propensity = self.df.PROPENSITY  # 取倾向性得分列
        groups = groups == groups.unique()[1]  # 标签列中0值改为False, 1改为Ture 
        n = len(groups)  # 样本总数量
        n1 = groups[groups==1].sum()  # 正样本个数
        n2 = n-n1  # 负样本个数
        g1, g2 = propensity[groups==1], propensity[groups==0]  # 取正样本、负样本的倾向性得分 

        if n1 > n2:  # 如果正样本数量大于负样本数量，那就把正负样本数量交换，得分也交换 ??? 
            n1, n2, g1, g2 = n2, n1, g2, g1

        # Randomly permute the treatment case IDs
        m_order = list(np.random.permutation(groups[groups==1].index))  # 取正样本索引，打乱后转化为list格式
        matches = {}  # 保存结果，形式为: 以正样本索引为key，对应的负样本索引组成的list为 value
        k = int(self.k)  # k的作用: 一个正样本匹配k个负样本

        # Match treatment cases to controls based on propensity score differences
        print("\nMatching [" + str(k) + "] controls to each case ... ", end = " ")
        for m in m_order:  # 循环遍历正样本索引 
            # Calculate all propensity score differences
            dist = abs(g1[m]-g2)  # 取当前正样本倾向性得分和所有负样本倾向性得分的差值
            array = np.array(dist)  
            # Choose the k smallest differences
            k_smallest = np.partition(array, k)[:k].tolist()  # 选择差值最小的前k个 
            if caliper:  # 如果在差值前三的基础上，要添加阈值，把阈值赋值给 caliper 就行 
                caliper = float(caliper)
                keep_diffs = [i for i in k_smallest if i <= caliper]
                keep_ids = np.array(dist[dist.isin(keep_diffs)].index)
            else:
                keep_ids = np.array(dist[dist.isin(k_smallest)].index)  # 取和当前样本差值最小的前三个样本的索引 

            # Break ties via random choice, if ties are present
            if len(keep_ids) > k:  # 如果 keep_ids 大于k，就从keep_ids中随机选3个索引，赋给matches中以正样本索引为key的value (这步基本不可能，因为前面已经限制了)
                matches[m] = list(np.random.choice(keep_ids, k, replace=False))
            elif len(keep_ids) < k:  # 如果keep_ids不足3个
                while len(matches[m]) <= k:  # 且matches中正样本的value长度小于k，用null补足
                    matches[m].append("NA")
            else:
                matches[m] = keep_ids.tolist()  # 正常情况: 把负样本索引组成的list赋给以当前正样本索引为key的记录作value

            # Matches are made without replacement
            if not replace:  # 是否允许不同正样本匹配相同负样本，如果不允许，就把已匹配的负样本索引从 g2 中删掉
                g2 = g2.drop(matches[m])

        # Prettify the results by consolidating into a DataFrame
        matches = pd.DataFrame.from_dict(matches, orient="index")
        matches = matches.reset_index()
        column_names = {}
        column_names["index"] = "CASE_ID"
        for i in range(k):
            column_names[i] = str("CONTROL_MATCH_" + str(i+1))
        matches = matches.rename(columns = column_names)

        # Extract data only for treated cases and matched controls
        matched_data = get_matched_data(matches, self.df)
        print("DONE!")
        write_matched_data(self.path, self.df)

        # Assign the matches and matched_data attributes to the Match object
        self.matches = matches
        self.matched_data = matched_data


    def evaluate(self, **kwargs):
        '''
        Conducts chi-square tests to verify statistically that the cases/controls
        are well-matched on the variables of interest.
        用卡方检验匹配的数据是否合适
        '''
        # Assert that the Match object has 'matches' and 'matched_data' attributes
        if not hasattr(self, 'matches'):
            raise AttributeError("%s does not have a 'matches' attribute." % (self))
        if not hasattr(self, 'matched_data'):
            raise AttributeError("%s does not have a 'matched_data' attribute." % (self))

        # Get variables of interest for analysis
        variables = self.df.columns.tolist()[0:-2]  # 取特征列名
        results = {}
        print("Evaluating matches ...")

        # Evaluate case/control match for each variable of interest
        for var in variables:  # 遍历每个列名 
            crosstable = make_crosstable(self.df, var)  # 行维是标签列，列维是当前遍历到的列名 var
            if len(self.df[var].unique().tolist()) <= 2:  # 判断列维可取值个数是不是小于等于2，调用不同的卡方计算方法
                p_val = calc_chi2_2x2(crosstable)[1]
            else:
                p_val = calc_chi2_2xC(crosstable)[1]
            results[var] = p_val  # 当前列的卡方值
            print("\t" + var, end = "")
            if p_val < 0.05:
                print(": FAILED")
            else:
                print(": PASSED")

        if True in [i < 0.05 for i in results.values()]:  # 最后判断特征列中，是否有特征列和标签列不相互独立
            print("\nAt least one variable failed to match!")  # 存在列和标签不相互独立 ? 
            return False
        else:
            print("\nAll variables were successfully matched!")
            return True


    def run(self, **kwargs):
        self.prepare_data()
        self.match()
        self.evaluate()



























































