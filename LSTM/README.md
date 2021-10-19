## 環境
- Google colab

## 重現方法
跟著以下步驟可以重現同樣成果：
1. [下載作業檔案](#下載作業檔案)
2. [準備資料](#準備資料)
3. [訓練model](#訓練model)

## 下載作業檔案
將LSTM.ipynb和DataPreparing.ipynb下載下來。

## 準備資料
前往[Kaggle](https://www.kaggle.com/c/digital-medicine-2021-case-presentation-1/data)  
Download All後解壓縮  
完成後擺放成以下結構:  
```
root_dir
  +- LSTM.ipynb
  +- DataPreparing.ipynb
  +- Case Presentation 1 Data
  |  +- Train_Textual
  |  |  +- ...txt
  |  +- Test_Intuitive
  |  |  +- ...txt
  |  +- Validation
  |  |  +- ...txt
```

### 訓練model
1. 按照上面結構擺好後，上傳到google drive
2. 點擊DataPreparing.ipynb，透過google colab開啟它
3. 在標題change your path here那裡修改your root path為你DataPreparing.ipynb所在的路徑
4. 點開上方執行階段 => 全部執行，就可以產生model需要的data
5. 接著點擊LSTM.ipynb，透過google colab開啟它
6. 點開左上角編輯 => 筆記本設定 => 選擇gpu => 儲存
7. 在標題change your path here那裡修改your root path為你LSTM.ipynb所在的路徑
8. 點開上方執行階段 => 全部執行，就可以得到預測結果的result.csv
