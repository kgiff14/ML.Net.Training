using DecisionTreeRegression.MachineLearning.Common;
using DecisionTreeRegression.MachineLearning.DataModels;
using DecisionTreeRegression.MachineLearning.Predictors;
using DecisionTreeRegression.MachineLearning.Trainers;

namespace DecisionTreeRegression
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new HousingData
            {
                Age = 65.2f,
                CrimeRate = 0.00632f,
                EmployCenterDistance = 4.0900f,
                HighwayAccessabilityRadius = 15.3f,
                NOConcentration = 0.538f,
                NumOfRoomsPerDwelling = 6.575f,
                Proportion = 2.31f,
                PTRatio = 15.3f,
                RiverCoast = 0,
                TaxRate = 296f,
                Zoned = 18f
            };

            var trainers = new List<ITrainerBase>
            {
                new DecisionTreeTrainer(3,3),
                new DecisionTreeTrainer(3,8, 0.1),
                new DecisionTreeTrainer(2,15),
                new DecisionTreeTrainer(5,2, 0.5),
                new DecisionTreeTrainer(2,46),
                new FastTreeTweedieTrainer(3,3),
                new FastTreeTweedieTrainer(3,8, 0.1),
                new FastTreeTweedieTrainer(2,15),
                new FastTreeTweedieTrainer(5,2, 0.5),
                new FastTreeTweedieTrainer(2,46),
                new GamTrainer()
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, HousingData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "boston_housing.csv";
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
            Console.WriteLine($"\nPrediction: {prediction.MedianPrice:#.##}");
        }
    }
}