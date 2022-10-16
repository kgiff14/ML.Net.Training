namespace RecommendationSystem.MachineLearning.DataModels
{
    public class MovieData
    {
        [LoadColumn(1)]
        public int MovieId;

        [LoadColumn(2)]
        public int UserId;

        [LoadColumn(3)]
        public float Label;
    }
}
