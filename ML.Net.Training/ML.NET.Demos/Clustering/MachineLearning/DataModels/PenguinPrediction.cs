namespace Clustering.MachineLearning.DataModels
{
    public class PenguinPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabelId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}
