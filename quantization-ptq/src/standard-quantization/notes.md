Metode Quantization 
 - torch.quantization.quantize_dynamic -> cpu only yang versi ringan 
 - model_8bit = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    load_in_8bit=True,     
    device_map="auto"     
) -> GPU

Dynammic PTQ udah cukup karna fokus researchnya gimana quantization ngaruh ke bobot internal modelnya dan performancenya bukan optimisasi spesifik deployment

| (testcase masih hardcode, masih pake model yang kecil, belum di finetuning) |
- researchnya bisa di bilang bestnya sih int8, latencynya paling gg dan akurasinya juga masih bagus. 
- fp16 malah kurang dibagian latency (malah makin lama). ini pas di browsing dapatnya mungkin karna hardware issue karna most of cpu itu ga bisa itung langsung fp16 tapi flownya fp16 -> convert ke fp32 -> hitung -> convert balik makanya maybe itu yang bikin agak lama

![alt text](images/first.png)

| (testcase dari dataset tweets INA) | 
788 Sample (1/30 dari datasets)
dengan config 
- num_inference_runs = 1
- warmup_runs = 5
====================================================================================================
QUANTIZATION COMPARISON SUMMARY (FP32 vs FP16 vs INT8)
====================================================================================================
             Metric FP32 (Baseline) FP16 (Half) INT8 (Quantized) FP16 vs FP32 INT8 vs FP32
    Model Size (MB)          474.79      237.43           230.15      +49.99%      +51.53%
       Accuracy (%)           75.00       75.00            74.75       +0.00%       -0.25%
 Avg Confidence (%)           88.16       88.16            88.13       -0.00%       -0.03%
  Mean Latency (ms)           66.54      150.35            54.70     +125.94%      -17.80%
Median Latency (ms)           58.47      143.79            55.30     +145.93%       -5.42%
   Std Latency (ms)           25.34       61.70            19.98          N/A          N/A
====================================================================================================

INT8 ada pengurangan accuracy sebesar 0.25% tapi latency lebih cepat 17.80%