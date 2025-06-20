# 2.4.0
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


# 2.0.0
データの整形のみ

# 2.2.0
重要なバグの修正：リックの開始時間位置が－１秒ずれていた

# 2.3.0
時間を同期する際のFP側の時刻は更新日時を参照することにした。
作成日時はずれることがあるため（すでにあるファイル名で実験を始めると、作成日時がもともとのファイルを参照される。削除しても）
またコピーしたファイルが使用できないため、オリジナルデータが削除されると二度とデータを復元できない
df/fからzscoreを算出していたが、蛍光から直接計算するように修正（論文がそうなっている）

# 2.4.0
更新日時は記録を終了した後、データの書き込みや保存が行われるため、12-13秒ほどタイムタグが発生する。
タイムタグは固定値ではないため、作成日時でゼロ合わせをする必要がある。
最初に削りすぎないほうがいいかもしれない？
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelmax
from scipy.integrate import trapz
from datetime import datetime
import glob
import os
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

def analyze_neural_activity(file_path_n, file_path_b, save_path): 
    
    df = pd.read_csv(file_path_n, encoding="shift-jis", skiprows =1) 

    # Step 0: pro proseccing
    # Delete data for the first 10 seconds
    df = df[df['Time(s)'] > 1]
    df = df.drop(df.index[-1])
    df = df.drop(df.columns[-1], axis=1)
    
    for i in range(len(df['Time(s)'])):
        if df.iloc[0]['AOut-2'] < 1:
            df = df.drop(df.index[0])
        elif df.iloc[0]['AOut-2'] > 0:
            break



    # AOut-2列が2回以上連続する場合の最初のインデックスと最後のインデックスを取得
    indices = np.where(df['AOut-2'].values[:-1] == 1)[0]  # 最後の要素を除外して1のインデックスを取得
    consecutive_indices_signal = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)  # 連続するインデックスのグループ化
    consecutive_indices_signal = [idx for idx in consecutive_indices_signal if len(idx) >= 2]  # 2つ以上の連続したインデックスのみ残す

    start_signal = [idx[0] for idx in consecutive_indices_signal]  # 最初のインデックス
    end_signal = [idx[-1] for idx in consecutive_indices_signal]  # 最後のインデックス


    # AOut-3列が2回以上連続する場合の最初のインデックスと最後のインデックスを取得
    indices = np.where(df['AOut-3'].values[:-1] == 1)[0]  # 最後の要素を除外して1のインデックスを取得
    consecutive_indices_ctrl = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)  # 連続するインデックスのグループ化
    consecutive_indices_ctrl = [idx for idx in consecutive_indices_ctrl if len(idx) >= 2]  # 2つ以上の連続したインデックスのみ残す

    start_ctrl = [idx[0] for idx in consecutive_indices_ctrl]  # 最初のインデックス
    end_ctrl = [idx[-1] for idx in consecutive_indices_ctrl]  # 最後のインデックス




    # Initialize lists to store new data
    new_signal_mean = []
    new_ctrl_mean = []
    new_Time_s_mean = []
    # Iterate over the selected sections
    for i in range(len(start_signal)-500):
        new_signal_mean.append(df.iloc[start_signal[i]:end_signal[i]]['gfp'].median()) 
        new_ctrl_mean.append(df.iloc[start_ctrl[i]:end_ctrl[i]]['tomato'].median())  
        new_Time_s_mean.append(df.iloc[start_signal[i]:end_ctrl[i]]['Time(s)'].mean())


    # Create a new DataFrame with the calculated values
    new_data = {
        'Time(s)': new_Time_s_mean,
        'Signal': new_signal_mean,
        'Control': new_ctrl_mean
    }

    syn_df = pd.DataFrame(new_data)
    
    
    #synchronized two data time
    #
    # Specify the range of rows to skip
    # Read the CSV file into a DataFrame
    df_b = pd.read_csv(file_path_b, encoding="shift-jis", skiprows=4, nrows=1)
    be = pd.to_datetime(df_b.iloc[0, 1],format='%Y/%m/%d %H:%M:%S')
    
    
    ####
        # Get the creation time of the file

    # Convert the timestamp to a datetime object
    # 更新日時を取得
    Modification_timestamp = os.path.getctime(file_path_n)
    ne = datetime.fromtimestamp(Modification_timestamp)
    be = be.replace(year=ne.year)
    zero_time = (be - ne).total_seconds()
    
    # Adjust the time values for the neural data
    syn_df['Time(s)'] = syn_df['Time(s)'] - zero_time
    syn_df = syn_df[syn_df['Time(s)'] < 2400]
    
    
    syn_df['Time(s)'] = syn_df['Time(s)'] + zero_time
    
    


    """ Step 1:Design the lowpass Butterworth filter
    """
    b, a = butter(2,10, btype='low',fs=50)
    smoothed_signal = filtfilt(b,a, syn_df['Signal'])
    smoothed_ctrl = filtfilt(b,a, syn_df['Control'])

    # Remove rows with NaN values
    valid_indices = ~np.isnan(smoothed_signal)
    smoothed_signal = smoothed_signal[valid_indices]
    time_seconds = syn_df['Time(s)'][valid_indices]
    smoothed_ctrl = smoothed_ctrl[valid_indices]
    
    
    # The double exponential curve we are going to fit.
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        '''Compute a double exponential function with constant offset.
        Parameters:
        t       : Time vector in seconds.
        const   : Amplitude of the constant offset. 
        amp_fast: Amplitude of the fast component.  
        amp_slow: Amplitude of the slow component.  
        tau_slow: Time constant of slow component in seconds.
        tau_multiplier: Time constant of fast component relative to slow. 
        '''
        tau_fast = tau_slow*tau_multiplier
        return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)

    
    max_sig = np.max(smoothed_signal)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    signal_parms, parm_cov = curve_fit(double_exponential, time_seconds, smoothed_signal, 
                                      p0=inital_params, bounds=bounds, maxfev=1500)
    signal_expfit = double_exponential(time_seconds, *signal_parms)

    # Fit curve to ctrl signal.

    max_sig = np.max(smoothed_ctrl)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    ctrl_parms, parm_cov = curve_fit(double_exponential, time_seconds, smoothed_ctrl, 
                                      p0=inital_params, bounds=bounds, maxfev=1500)
    ctrl_expfit = double_exponential(time_seconds, *ctrl_parms)
     

    def linear_function(t, slope, intercept):

        return slope * t + intercept
    
    """
    max_ctrl = np.max(smoothed_ctrl)
    initial_params = [0, max_ctrl / 2]  # 初期パラメータ: [傾き, 切片]
    bounds = ([-np.inf, 0], [np.inf, max_ctrl])  # 境界: 傾きは無制限、切片は0からmax_ctrlまで
    ctrl_params, parm_cov = curve_fit(linear_function, time_seconds, smoothed_ctrl,
                                      p0=initial_params, bounds=bounds, maxfev=1500)
    ctrl_expfit = linear_function(time_seconds, *ctrl_params)
    """

    signal_detrended = smoothed_signal - signal_expfit
    ctrl_detrended = smoothed_ctrl - ctrl_expfit
    
 
    
    """ Step 2: Fitting control signal to signal signal
        Perform linear regression on the control data
    """
    slope, intercept, r_value, p_value, std_err = linregress(ctrl_detrended, signal_detrended)
    signal_est_motion = intercept + slope * ctrl_detrended
    signal_corrected = signal_detrended - signal_est_motion
    
    """ Step 3: Calculate dF/F using the function
        Calculate dF/F
    """
    df_f_signal = 100*signal_corrected/signal_expfit
    # (smoothed_signal - signal_expfit - signal_est_motion) / signal_expfit
    
    # Step 4: Standard data 
    Zscore_signal = (signal_corrected - np.mean(signal_corrected)) / np.std(signal_corrected)
    
    
    # 
    time_seconds = time_seconds - zero_time
    

    """ Behavioral Data Analysis
        Step 5: caluculate Burst 
    """ 

    
    
    processed_data = pd.DataFrame({
    'Time(s)': time_seconds,
    'dF/F': df_f_signal,
    'Z-score': Zscore_signal
    })
    # Save the processed data to a CSV file
    file_name = os.path.splitext(os.path.basename(file_path_n))[0]
    save_path = os.path.join(save_path, f'processed_{file_name}.csv')
    processed_data.to_csv(save_path, index=False)
    
    
#    return None 戻り値が必要のない場合はこのコードがあってもなくても同じ結果となる