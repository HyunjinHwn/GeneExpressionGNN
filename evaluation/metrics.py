import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import pandas as pd
import pickle
import torch
from .utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))

# ================================= GT, lmid load  ==================================
lmid_path = os.path.join(current_dir, "metadata/lmid_970.txt")
lmid_index = {}
with open(lmid_path, 'r') as f: 
    lmid = [l.strip() for l in f.readlines()]
    
lmid_index[('order', 970)] = lmid
lmid_index[('order', 970, 1)] = lmid
lmid_index[('order', 970, 2)] = lmid
lmid_index[('order', 486)] = lmid_index[('order', 970)][:486]
lmid_index[('order', 324)] = lmid_index[('order', 970)][:324]
lmid_index[('order', 108)] = lmid_index[('order', 970)][:108]
lmid_index[('order', 10)] = lmid_index[('order', 970)][:10]
lmid_index[('order', 1)] = lmid_index[('order', 970)][:1]
lmid_index[('lasso', 108)] = ['5997', '5054', '23670', '7466', '3385', '56654', '5696', '2185',
                            '3315', '8553', '6616', '7849', '7494', '991', '6347', '6813',
                            '11065', '3337', '3157', '25932', '10398', '4860', '1050', '10776',
                            '949', '7168', '79170', '10610', '9455', '5480', '6275', '6915',
                            '7867', '1759', '54541', '348', '29763', '1831', '3202', '56924',
                            '2690', '3566', '3815', '5788', '27346', '5357', '6195', '11261',
                            '23530', '1277', '1958', '4582', '1052', '4616', '8851', '6696',
                            '2353', '4609', '3775', '3206', '2064', '2274', '3800', '3162',
                            '1282', '1026', '11151', '10797', '976', '2625', '10857', '9289',
                            '4783', '9805', '2920', '11098', '56997', '823', '6812', '3693',
                            '3122', '5909', '1123', '3486', '5603', '9133', '5836', '960',
                            '2065', '7074', '664', '9531', '230', '7106', '1978', '7852',
                            '5777', '84617', '2745', '3280', '5236', '26227', '8870', '2810',
                            '10276', '4638', '481', '291']
lmid_index[('greedy_forward', 108)] = ['3800','4303','7168','2810','5641','813','291','3930',
                            '256364','5909','1759','873','6599','10644','2037','976',
                            '230','949','79073','10610','10904','8480','11230','5108',
                            '1514','664','11065','2274','22883','9467','9289','10276',
                            '27346','3628','10857','1666','6195','644','6919','26227',
                            '5836','4651','8553','7494','642','11041','4043','23585',
                            '3206','23670','2058','5054','22889','9926','3122','4783',
                            '4925','3693','2896','1452','3988','5092','6275','51097',
                            '25805','25987','25932','9961','23077','10797','5423','3337',
                            '2064','481','5782','727','3925','1831','6499','58472',
                            '4860','23300','1153','2353','11344','6603','9246','8870',
                            '4927','8678','9868','6659','5154','25803','7994','3815',
                            '7296','3566','5829','10237','11188','5525','622','7867',
                            '2195','54541','9181','1647']
lmid_index[('greedy_forward', 108, 1)] = ['3800','3385','5603','9697','7168','23530','5909','4927',
        '3108','5829','2810','27346','11065','2058','3930','5836',
        '23077','25932','7849','4925','11230','3033','3206','10013',
        '4651','664','9276','2745','873','5110','1831','2274',
        '642','8553','6915','8985','10857','2037','230','2195',
        '29937','9289','51097','10915','10276','10237','11261','7494',
        '9531','1759','596','5641','25987','23670','7982','10953',
        '9170','10797','3551','4016','348','25793','10099','4860',
        '9221','26227','1514','644','5547','3815','2353','7159',
        '10681','622','1956','9805','5106','8624','51031','9181',
        '2184','66008','9761','10904','54733','25805','2771','5359',
        '3337','5092','5997','6804','27032','1282','1026','3988',
        '4303','23585','9455','1445','5355','54541','23588','16',
        '10493','291','1647','8870']
lmid_index[('greedy_forward', 108, 2)] = ['3800','3385','2810','7168','291','6195','6275','4303',
        '5909','10681','2037','3930','8480','23585','27346','23670',
        '949','1759','2274','7849','664','10276','873','4925',
        '644','9467','392','3206','1514','10904','10610','26227',
        '4651','22889','7074','6810','10797','7494','7153','8553',
        '5836','6599','25932','2624','4043','25966','4016','11041',
        '5468','4783','5092','23077','1666','3628','79073','10237',
        '3098','25805','7158','622','9289','22883','6919','6603',
        '4860','7982','3815','596','3337','3108','2958','9170',
        '642','230','23530','8624','3156','10099','9697','1026',
        '2058','2582','5423','9019','51031','5782','3566','6284',
        '1153','310','57406','1831','10953','147179','9805','2353',
        '27032','10857','2745','51005','9448','9961','7852','66008',
        '10013','55556','3775','10057']
lmid_index[('greedy_forward', 54)] = lmid_index[('greedy_forward', 108)][:54]

def performance_all(prediction_path, model_name, feature_selection_method, num_landmarks, root=''):
    output = [prediction_path, model_name, feature_selection_method, num_landmarks]
    
    # ================================= INF load  ==================================
    with open(os.path.join(current_dir, "metadata/geneid_12320.txt"), "r") as f:
        rid = [line.strip() for line in f.readlines()]
    index_all = pd.Index(rid, name='rid')
    lmid = lmid_index[(feature_selection_method, num_landmarks)]
    GT_path = f"{root}GT.pkl"
    with open(GT_path, 'rb') as f:
        GT = pickle.load(f)   
    print(f"model:{model_name}, num(lm):{len(lmid)}")
    try:
        INF = torch.load(root+prediction_path) # numpy
    except:
        with open(root+prediction_path, 'rb') as f:
            INF = pickle.load(f)   
    if type(INF) == list:
        INF = np.array(INF).T.squeeze()
    if INF.shape[0] == 176:
        INF = INF.T
    # ================================= err, scc, pcc  ==================================
    err = calculate_Error(GT, INF)
    scc = calculate_SCC(GT, INF)
    pcc = calculate_PCC(GT, INF)
    output += [f"{np.mean(err):.4f}",f"{np.std(err):.4f}", f"{np.mean(scc):.4f}",f"{np.std(scc):.4f}", 
            f"{np.mean(pcc):.4f}",f"{np.std(pcc):.4f}"] 
    
    # ================================= recall, BING  ==================================
    r95, r99, BING_df = recall_for_LM(GT, INF, lmid, index_all)
    output += [r95, r99]

    # ================================= level4  ==================================
    level4_df_GT = level3_to_level4(GT, rid)
    level4_df = level3_to_level4(INF, rid)
    
    # ================================= gene rank  ==================================
    ranked_INF_df_GT = level4_df_GT.rank(axis=0)
    ranked_INF_df = level4_df.rank(axis=0)
    del INF
    
    # ================================= gene rank correlation  ==================================
    # compare ranks in AIG
    corr_all = []
    for i in range(GT.shape[1]):
        gti = ranked_INF_df_GT.iloc[:,i].to_numpy()
        infi = ranked_INF_df.iloc[:,i].to_numpy()
        corr_all.append(np.corrcoef(gti, infi)[0,1])

    # compare ranks in BING
    bing_ids = BING_df[BING_df.iloc[:, 4] == 1].iloc[:, 1].astype(str).tolist()
    GT_bing = ranked_INF_df_GT.loc[bing_ids,:]
    GT_bing = GT_bing.rank(axis=0)
    GT_bing = np.array(GT_bing)
    INF_bing = ranked_INF_df.loc[bing_ids,:]
    INF_bing = INF_bing.rank(axis=0)

    corr_bing = []
    for i in range(GT_bing.shape[1]):
        gti = GT_bing[:,i]
        infi = INF_bing.iloc[:,i].to_numpy()
        corr_bing.append(np.corrcoef(gti, infi)[0,1])
        
    output += [GT.shape[0], f"{np.mean(corr_all):.4f}",f"{np.std(corr_all):.4f}",
                GT_bing.shape[0], f"{np.mean(corr_bing):.4f}",f"{np.std(corr_bing):.4f}"]
    return output