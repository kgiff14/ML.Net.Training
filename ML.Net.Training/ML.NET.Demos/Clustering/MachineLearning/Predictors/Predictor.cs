using Clustering.MachineLearning.DataModels;

namespace Clustering.MachineLearning.Predictors
{
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "ClusterModel.mdl");
        private readonly MLContext _mlContext;
        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(11);
        }

        public PenguinPrediction Predict(PenguinData sample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<PenguinData, PenguinPrediction>(_model);

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
