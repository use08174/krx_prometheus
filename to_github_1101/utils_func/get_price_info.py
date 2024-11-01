import os
import numpy as np
import pandas as pd

price_delta_pct = 3
current_dir = os.path.dirname(__file__)

def transform_data(df):
    # 날짜를 datetime 형식으로 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜').reset_index(drop=True)
    
    # 전일 대비 퍼센트 변화율 계산
    df['open'] = df['시가'].pct_change() * 100
    df['high'] = df['고가'].pct_change() * 100
    df['low'] = df['저가'].pct_change() * 100
    df['close'] = df['종가'].pct_change() * 100
    df['adj-close'] = df['close']  # Adjusted close 설정 (기본적으로 동일)

    # 상승값 계산 함수 정의
    def calculate_increase(df, days):
        return df['종가'].pct_change(periods=days) * 100

    # 상승값 컬럼 추가 (5일, 10일, 15일, 20일, 25일, 30일 전 대비 상승률)
    for days in [5, 10, 15, 20, 25, 30]:
        df[f'inc-{days}'] = calculate_increase(df, days)
    
    # 필요한 열 선택 및 이름 재정렬
    df = df[['날짜', 'open', 'high', 'low', 'close', 'adj-close', 
             'inc-5', 'inc-10', 'inc-15', 'inc-20', 'inc-25', 'inc-30']]
    
    # 컬럼명 및 날짜 포맷 재설정
    df.rename(columns={'날짜': 'date'}, inplace=True)
    df = df.round(2)  # 소수점 둘째 자리까지 반올림

    return df

def df_to_markdown_table(df):
    # 열 이름을 문자열로 변환
    header = "|    ****| " + " | ".join([f"**{col}**" for col in df.columns]) + " |"
    separator = "|---:" + "|".join(["-----------:" for _ in df.columns]) + "|"
    
    # 데이터 행을 문자열로 변환
    rows = []
    for idx, row in df.iterrows():
        row_str = f"|  {idx} | " + " | ".join([f"{row[col]:>8}" if isinstance(row[col], (int, float)) else f"{row[col]}" for col in df.columns]) + " |"
        rows.append(row_str)
    
    # 최종 문자열 생성
    table_str = "\n".join([header, separator] + rows)
    return table_str

def get_price_external_info():
    # 기본 경로 설정
    path = '../price_source_folder/'
    path = os.path.join(current_dir, path)
    print(path)

    # 랜덤 티커 고르기
    price_ticker_list = [f for f in os.listdir(path) if f != '.DS_Store']
    random_ticker_idx = int(np.random.uniform(0, len(price_ticker_list)))
    random_ticker = price_ticker_list[random_ticker_idx]

    # 랜덤 날짜 설정 (30일 전 기준으로 비교할 수 있도록 이전 날짜도 선택)
    date_path = os.path.join(path, random_ticker)
    date_list = sorted([f for f in os.listdir(date_path) if f != '.DS_Store'])
    random_date_idx = int(np.random.uniform(1, len(date_list)-1))  # 최소 1부터 시작해 이전 데이터 확보
    random_date = date_list[random_date_idx]
    prev_random_date = date_list[random_date_idx - 1]
    next_random_date = date_list[random_date_idx + 1]

    # 각각의 CSV 경로 설정
    csv_folder_path_current = os.path.join(date_path, random_date)
    csv_name_current = os.listdir(csv_folder_path_current)[0]
    csv_path_current = os.path.join(csv_folder_path_current, csv_name_current)

    csv_folder_path_prev = os.path.join(date_path, prev_random_date)
    csv_name_prev = os.listdir(csv_folder_path_prev)[0]
    csv_path_prev = os.path.join(csv_folder_path_prev, csv_name_prev)

    # CSV 불러오기
    df_current = pd.read_csv(csv_path_current)
    df_prev = pd.read_csv(csv_path_prev)

    # 두 DataFrame 합치기
    df_combined = pd.concat([df_prev, df_current], ignore_index=True)

    # 데이터 변환 및 결측값 제거
    transformed_df = df_to_markdown_table(transform_data(df_combined).dropna())

    # 현재 종가를 기준으로 상승/하락 판별을 위한 초기 변수 설정
    current_close = df_current['종가'].iloc[-1]
    change_detected = False
    change_type = []

    # 다음 날짜의 데이터가 조건을 만족할 때까지 while문 반복
    while not change_detected:
        # 다음 날짜 CSV 경로 설정
        csv_folder_path_next = os.path.join(date_path, next_random_date)
        csv_name_next = os.listdir(csv_folder_path_next)[0]
        csv_path_next = os.path.join(csv_folder_path_next, csv_name_next)
        
        # 다음 날짜 데이터 불러오기
        df_next = pd.read_csv(csv_path_next)
        
        # 다음 날짜 종가를 기준으로 상승/하락 계산
        next_close = df_next['종가'].iloc[-1]
        change_percentage = ((next_close - current_close) / current_close) * 100

        # 상승/하락 판별
        if change_percentage >= price_delta_pct:
            change_type.append('상승')
            change_detected = True
        elif change_percentage <= -price_delta_pct:
            change_type.append('하락')
            change_detected = True
        else:
            # 변동폭이 조건을 만족하지 않을 경우 다음 날짜로 업데이트
            random_date_idx += 1
            if random_date_idx + 1 < len(date_list):
                next_random_date = date_list[random_date_idx + 1]
            else:
                print("데이터가 부족하여 조건을 만족하는 변동을 찾지 못했습니다.")
                break

    final_dict = {
        'table': transformed_df,
        'change_type': change_type[0] if change_type else None
    }

    return final_dict
