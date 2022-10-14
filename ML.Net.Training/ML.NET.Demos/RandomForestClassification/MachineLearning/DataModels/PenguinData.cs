namespace RandomForestClassification.MachineLearning.DataModels
{
    public class PenguinData
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(2)]
        public float CulmenLength { get; set; }

        [LoadColumn(3)]
        public float CulmenDepth { get; set; }
    }
}
