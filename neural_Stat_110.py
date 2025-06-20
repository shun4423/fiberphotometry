# 1.1.0
"""
time division用(Linearモード)に前処理としてコードを追加
Fitting用のbaselineを設定するにともなって、
Fittingおよびベースライン期間を除くDf/fの計算を
.Doricから.csvに変更に伴って、読み込みのコードを変更
交互に照射されるため、ワンサイクルをひとまとめにする。
ごくまれに照射のラグがあるのか照射されても光らないことがあるため、平均値ではなく、中央値としてまとめた
zscoreピークのカットオフ値を2.5に設定
timeというラベルをややこしいので元のTime(s)に変更
時間は神経データをベースに行動データを補正していたがそれをやめた。
つまり、行動データを基準に神経データを補正し、０時間のときに最初のリックがあることを示す。



神経データの整形と統計的な解析を別のコードに分けることにした。これは、追加サンプルをふくめた解析をやりやすくするため

# 1.1.0
stop_lickの計算に"-1"があったが、不要なので削除
IBI区間を実装
First_peakは論文を参考に30秒に変更

#　置換用　Z-score, Df_signal
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelmax
from scipy.integrate import trapz
from scipy.stats import linregress
from datetime import datetime
import glob
import os
import seaborn as sns
from scipy.fft import fft
from scipy.signal import medfilt, butter, filtfilt
from scipy.optimize import curve_fit, minimize

def stat_neural_activity(file_path_n, file_path_b, save_path, fig_out): 
    test_zero_time = 0
    test_time = 300
    
    
    df_n = pd.read_csv(file_path_n, encoding="shift-jis")
    Df_signal = df_n['Normalized'] #Normalized, Z-score
    time_sec = df_n['Time(s)']
    downsampling_factor = 1
    # Downsample the signal by selecting every 'downsampling_factor'-th data point for easy fig
    #Df_signal = Df_signal.groupby(Df_signal.index // downsampling_factor).mean()
    #time_sec = time_sec.groupby(time_sec.index // downsampling_factor).mean()
    
    # Specify the range of rows to skip
    skiprows = list(range(16))  # Skip rows 1 through 16
    # Read the CSV file into a DataFrame
    df_b = pd.read_csv(file_path_b, encoding="shift-jis", skiprows=skiprows)
    df_val =  df_b.iloc[:,4].values
    
    # Decide on the data partition to use 
    #
    Lick_end_block = 1 * 60 * 15 # 集計間隔 * 60sec * min
    df_val = df_val[:Lick_end_block]

    
    """ Behavioral Data Analysis
        Step 5: caluculate Burst 
    """ 
    start_lick, time_list, IBI_list, j=[], [], [], []
    time_list.clear()
    IBI_list.clear()
    pre_burst_end=0


    def get_burst(arr, N):
        # 最初に現れた0でない値から、その次の0がN連続するまでの間を取得
        
        arr = np.r_[arr, np.zeros(N, dtype=int)]#最後にゼロを追加：ゼロがn回続けば一つの塊として処理するため、続かない場合のエラーを回避する
        is_nonzero = arr != 0# 0=F, 0>Tとすることで0>の区別を消す
        # 始点＝最初に現れた0以外の値の位置(リスト内が論理演算のため、TRUEが大きく一番先に検出したTを返す)
        #そのためには、bool型をNumPy配列型に変換する必要がある（EX.　np.array(bool_data)）。
        s = is_nonzero.argmax()
        ln = np.convolve(is_nonzero[s:], np.ones(N, dtype=int), 'valid').argmin()# sより後ろで初めて0がN連続する箇所のindex
        #備考：論理演算子に対してconvolve関数（第２引数が１）でフィルタするとTを１に、Fを０にすることができる


        return arr[s:s+ln], (s, s+ln), s, ln#クラスターを返す、クラスターの開始時間マス、クラスターの時間, interBintervl


    First_lick_flag = False 
    for i in range(len(df_val)):
        out, [s, e], st_l, time= get_burst(df_val, 1)# この1はpause criteriaの1秒
        j.append(out.sum())#リック内の０も含むがmeanでなくsumなので問題なし
        time_list.append(time)#0始まりのindexが返される（-1すべきかもしれないがIBIの秒数に間違いはない）
        start_lick.append(st_l)#0始まりのindexが返されることで、正しい開始時刻に補正される。  
        """ 補正される理由：なめはじめた時間を0とするため。
            indexをなめてからの経過時間としてそのまま利用している
        """
        df_val[:e] = 0
        if pre_burst_end > 0 and (out.sum() != 1):
            post_burst_str= st_l+1
            IBI_list.append(post_burst_str-pre_burst_end)
        if df_val.sum() == 0:
            break              
        if out.sum() != 1:
            pre_burst_end= st_l+1+time
    
    
    stop_lick, start_IBI,stop_IBI = [],[],[]
    for i in range(len(start_lick)):
        stop_lick.append(start_lick[i] + time_list[i])
        
    for i in range(len(IBI_list)):
        stop_IBI.append(start_lick[i] + time_list[i] + IBI_list[i])
        
        
    average = np.mean(j)
    count = len(j)
    IBI = np.mean(IBI_list)





    """ Step 7: Finding peaks
    """ 
    AUC_lick, AUC_IBI =[],[]
    positive_signal = Df_signal
    # positive_signal = np.maximum(Df_signal, 0)
    for i in range(len(start_lick)):
        start_time = start_lick[i]
        end_time = stop_lick[i]
        licking_time = end_time - start_time 
        index_lick = np.where((time_sec >= start_time) & (time_sec <= end_time))[0]
        area = np.trapz(positive_signal[index_lick], time_sec[index_lick])
        AUC_lick.append(area/licking_time)
    
    for i in range(len(stop_IBI)):
        start_time = stop_lick[i]
        end_time = stop_IBI[i]
        IBI_time = end_time - start_time
        index_lick = np.where((time_sec >= start_time) & (time_sec <= end_time))[0]
        area = np.trapz(positive_signal[index_lick], time_sec[index_lick])
        AUC_IBI.append(area/IBI_time)
    
    index_lick_zs = np.where((time_sec >= 0) & (time_sec <= 60))[0]
    zs_AUC_1min = np.trapz(Df_signal[index_lick_zs], time_sec[index_lick_zs])
    
    index_lick_zs_5 = np.where((time_sec >= test_zero_time) & (time_sec <= 300))[0]
    zs_AUC_5min = np.trapz(Df_signal[index_lick_zs_5], time_sec[index_lick_zs_5])
    
    
    index_lick_zs_10 = np.where((time_sec >= test_zero_time) & (time_sec <= 600))[0]
    zs_AUC_10min = np.trapz(Df_signal[index_lick_zs_10], time_sec[index_lick_zs_10])
    
    index_lick_zs_15 = np.where((time_sec >= test_zero_time) & (time_sec <= 900))[0]
    zs_AUC_15min = np.trapz(Df_signal[index_lick_zs_15], time_sec[index_lick_zs_15])

    M_AUC_lick = np.mean(AUC_lick)
    M_AUC_IBI=np.mean(AUC_IBI)
    
    
    # order：中心点に対して左右のオーダー点（これが指定する値）。このなかから相対最大値を調べる
    peaks_indices = eaks_indices = argrelmax(Df_signal.values, order=100)[0] 
    
    # Apply a threshold to filter peaks
    threshold = 0  # Adjust the threshold as needed
    peaks_indices = peaks_indices[Df_signal.iloc[peaks_indices] > threshold]
    
    # Set the threshold value
    threshold_value = 2
    thres_peak_index = peaks_indices[Df_signal.iloc[peaks_indices] > threshold_value]
    
    
    # Define the time of the first peak
    start_time_f = start_lick[0] # 0
    time_tolist = time_sec.tolist()
    
    # Creat dataFlame
    # threshold = 2
    df_peaks_thres= pd.DataFrame(index=[], columns=['time','value','peaks'])
    df_peaks_thres['value'] =  Df_signal # set orignal data
    df_peaks_thres['peaks'] = Df_signal.iloc[thres_peak_index] # identify found peaks area
    df_peaks_thres['value'] =  Df_signal # re-set orignal data
    df_peaks_thres['time'] = time_sec # set time data  
    df_peaks_thres = df_peaks_thres[df_peaks_thres['time'] >= start_time_f]
    df_thres = df_peaks_thres.dropna(subset=['peaks']).copy()
    interval_mean_thres2 = df_thres['time'].diff().dropna().mean()
    mean_value_thres2 = df_thres['peaks'].mean()
    count_value_thres2 = df_thres['peaks'].count()

    
    

    # Creat dataFlame
    # threshold = 0
    df_peaks= pd.DataFrame(index=[], columns=['time','value','peaks'])
    df_peaks['value'] =  Df_signal # set orignal data
    df_peaks['peaks'] = Df_signal.iloc[peaks_indices] # identify found peaks area
    df_peaks['value'] =  Df_signal # re-set orignal data
    df_peaks['time'] = time_sec # set time data
    

    
    #Filter DF by first peak time
    filtered_df = df_peaks[(df_peaks['time'] >= test_zero_time) & (df_peaks['time'] <= test_time)].copy()
    filtered_df = filtered_df.dropna(subset=['peaks']) # for intensity
    interval_mean_thres0 = filtered_df['time'].diff().dropna().mean()
    interval_list = np.diff(np.diff(time_sec[peaks_indices]))
    
    interval_counts = []
    interval_counts.append(len(interval_list <= 0.5))

    Mean_sig_3s = df_n.loc[(df_n['Time(s)'] < 3) & (df_n['Time(s)'] > 0), 'Z-score'].max()
    Mean_sig_30s = df_n.loc[(df_n['Time(s)'] < 30) & (df_n['Time(s)'] > 0), 'Z-score'].mean()

    
    

    # Calculate the average intensity of the peaks
    average_peak_intensity = np.mean(filtered_df['peaks'])

    # Calculate the number of peaks per minute based on the total measurement time
    peaks_number = len(filtered_df['peaks'])
    
    


    # Create a mask for values exceeding the threshold
    # mask = positive_signal > threshold_value    
    
    
    # Time of max signal
    max_index = df_n.loc[df_n['Time(s)'] >= 0, 'Z-score'].idxmax()
    max_value_time = df_n.loc[max_index, 'Time(s)']
    
    
    # Plot all peaks on the signal for the entire period
    upper=np.max(Df_signal.values)
    lower=np.min(Df_signal.values)
    file_name = os.path.splitext(os.path.basename(file_path_n))[0]
    if fig_out == True:
        plt.figure(figsize=(10, 6))
        plt.plot(time_sec, Df_signal, color='black', label='Signal', alpha = 0.5)  # Plot signal in black
        plt.plot(filtered_df['time'], filtered_df['peaks'], 'r.') 
        for i in range(len(start_lick)):
            start_time = start_lick[i]
            end_time = stop_lick[i]
            time_tolist = time_sec.tolist()
            start_index = time_tolist.index(time_sec[time_sec>start_time].iat[0])
            end_index = time_tolist.index(time_sec[time_sec>end_time].iat[0])
            plt.fill_between(time_sec.iloc[start_index:end_index].values,upper,lower , color='lightcoral', alpha=0.3)
        plt.title('')
        plt.xlabel('Time (sec)', fontsize=18)
        plt.ylabel('Z-score', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        #plt.ylabel('Df/f (%)')
        image_save_path = os.path.join(save_path, f'3_{file_name}_plot.png')
        plt.savefig(image_save_path, format='png')
        # plt.savefig(image_save_path)
        plt.close()  # Close the plot to avoid displaying it


    
    # Return the results
    return (file_name, 
            average, 
            count,
            IBI,
            average_peak_intensity, 
            peaks_number, 
            upper, 
            max_value_time,
            Mean_sig_3s,
            Mean_sig_30s,
            zs_AUC_1min,
            zs_AUC_5min,
            zs_AUC_10min,
            zs_AUC_15min,
            interval_mean_thres0,
            interval_mean_thres2,
            mean_value_thres2,
            count_value_thres2,
            M_AUC_lick,
            M_AUC_IBI,
            interval_counts)