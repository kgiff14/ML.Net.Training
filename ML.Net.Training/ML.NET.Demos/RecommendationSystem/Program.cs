using RecommendationSystem.MachineLearning.Common;
using RecommendationSystem.MachineLearning.DataModels;
using RecommendationSystem.MachineLearning.Predictors;
using RecommendationSystem.MachineLearning.Trainers;

namespace RecommendationSystem
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new MovieData
            {
                UserId = 148844,
                MovieId = 1
            };

            var trainers = new List<ITrainerBase>
            {
                new  MatrixFactorizationTrainer(100, 50),
                new  MatrixFactorizationTrainer(200, 150),
                new  MatrixFactorizationTrainer(100, 50, 0.3),
                new  MatrixFactorizationTrainer(120, 5),
                new  MatrixFactorizationTrainer(10, 500),
                new  MatrixFactorizationTrainer(10, 5, 0.5),
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, MovieData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "netflix_subset.csv";
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Data\", fileName);

            trainer.Fit(path.ToString());

            var modelMetrics = trainer.Evalute();

            Console.WriteLine($"\nLoss Function: {modelMetrics.LossFunction: 0.##}");
            Console.WriteLine($"\nMean Absolute Error: {modelMetrics.MeanAbsoluteError: #.##}");
            Console.WriteLine($"\nMean Squared Error: {modelMetrics.MeanSquaredError: #.##}");
            Console.WriteLine($"\nRSquared: {modelMetrics.RSquared: 0.##}");
            Console.WriteLine($"\nRoot Mean Squared Error: {modelMetrics.RootMeanSquaredError: 0.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\nPrediction: {prediction.Score}");
        }
    }
}