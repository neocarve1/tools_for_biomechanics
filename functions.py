"""
데이터를 기준점으로 자르고 resampling 하는 함수

input으로 raw_data와 heel_strike 지점 그리고 원하는 resample_dimension을 넣으면 정리된 값이 나온다. 

예시)
data.shape => (36123,)
points.shape => (530,)
resample_dim = 100
flatten=False

그러면 output.shape => (530, 100) 이 나온다. 

flatten=True 로 하면 output.shape => (530*100,) 이 나온다. 
"""

def cut_data_by_cycle_and_resample(data, points, resample_dim, flatten=False):
    resampled_data = np.zeros((len(points)-1, resample_dim))

    for i in range(1, len(points)):
        tmp = np.array(data[points[i-1]:points[i]])
        resampled_tmp = np.zeros(((len(tmp)-1)*resample_dim))
        
        for j in range((len(tmp)-1)*resample_dim):
            if j % resample_dim == 0:
                resampled_tmp[j] = tmp[j//resample_dim]
            else:
                resampled_tmp[j] = np.nan

        resampled_tmp = pd.Series(resampled_tmp)
        resampled_tmp.iloc[-1] = tmp[-1]

        resampled_tmp = resampled_tmp.interpolate(method='cubic')
        resampled_tmp = np.array(resampled_tmp)

        for j in range(len(resampled_tmp)-1, -1, -1):
            if j % (len(tmp)-1) != 0:
                resampled_tmp = np.delete(resampled_tmp, j, axis=0)
        
        resampled_data[i-1, :] = resampled_tmp
    
    if flatten == False:
        return resampled_data
    else:
        return resampled_data.reshape(-1)
        

"""
input으로 raw_data와 heel_strike 지점, 원하는 resample_dimension과 뽑고 싶은 값을 넣으면 미분된 값이 resample 되어서 나온다. 

예시)
data.shape => (36123,)
points.shape => (530,)
resample_dim = 100
flatten=False

그러면 output.shape => (530, 100) 이 나온다. 

order의 
0번째 index는 도함수
1번째 index는 이계도함수
2번째 index는 삼계도함수
를 의미한다. 

셋 모두를 뽑고 싶으면 order=[1, 1, 1]
도함수와 삼계도함수만 뽑고 싶으면 order=[1, 0, 1]
이런식으로 입력하면 된다.
"""

def differentiate_data_and_resample(data, points, sampling_rate, resample_dim, order = [1, 1, 1]):
    if order[0] == 1:
        resampled_first_derivative = np.zeros((len(points)-1, resample_dim))
    if order[1] == 1:
        resampled_second_derivative = np.zeros((len(points)-1, resample_dim))
    if order[2] == 1:
        resampled_third_derivative = np.zeros((len(points)-1, resample_dim))

    for i in range(len(points)-1):
        tmp = np.array(data[points[i]:points[i+1]])
        first = (tmp[1:] - tmp[:-1])/sampling_rate
        second = (first[1:] - first[:-1])/sampling_rate
        third = (second[1:] - second[:-1])/sampling_rate
        if order[0] == 1:
            resampled_first_derivative[i] = cut_data_by_cycle_and_resample(first, [0, len(first)], 100, flatten=True)
        if order[1] == 1:
            resampled_second_derivative[i] = cut_data_by_cycle_and_resample(second, [0, len(second)], 100, flatten=True)
        if order[2] == 1:
            resampled_third_derivative[i] = cut_data_by_cycle_and_resample(third, [0, len(third)], 100, flatten=True)

    result = []
    if order[0] == 1:
        result.append(resampled_first_derivative)
    if order[1] == 1:
        result.append(resampled_second_derivative)
    if order[2] == 1:
        result.append(resampled_third_derivative)

    return result


