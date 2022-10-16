namespace RecommendationSystem.MachineLearning.DataModels
{
    public class MoviePredition
    {
        [ColumnName("Label")]
        public float Label;

        [ColumnName("Score")]
        public float Score;
    }
}
