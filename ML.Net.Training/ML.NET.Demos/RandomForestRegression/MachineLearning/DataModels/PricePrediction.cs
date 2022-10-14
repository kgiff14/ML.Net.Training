namespace RandomForestRegression.MachineLearning.DataModels
{
    public class PricePredictions
    {
        //Probability/estimations of continous values are found in Score column - ML.Net
        [ColumnName("Score")]
        public float MedianPrice { get; set; }
    }
}
