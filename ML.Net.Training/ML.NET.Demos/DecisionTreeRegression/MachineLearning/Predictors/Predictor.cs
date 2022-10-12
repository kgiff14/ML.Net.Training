using DecisionTreeRegression.MachineLearning.DataModels;

namespace DecisionTreeRegression.MachineLearning.Predictors
{
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "DecisionTreeRegressionModel.mdl");
        private readonly MLContext _mlContext;
        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(11);
        }

        public PricePredictions Predict(HousingData sample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HousingData, PricePredictions>(_model);

            return predictionEngine.Predict(sample);
        }

        private void LoadModel()
        {
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException(ModelPath);
            }

            using var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read);

            _model = _mlContext.Model.Load(stream, out _);

            if (_model is null)
            {
                throw new Exception("Failed to load model");
            }
        }
    }
}
