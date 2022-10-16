using RecommendationSystem.MachineLearning.DataModels;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace RecommendationSystem.MachineLearning.Common
{
    public class TrainerBase : ITrainerBase
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "RecommendationModel.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<MatrixFactorizationPredictionTransformer, MatrixFactorizationModelParameters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(111);
        }

        public RegressionMetrics Evalute()
        {
            var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);

            return MlContext.Regression.Evaluate(testSetTransform);
        }

        public void Fit(string fileName)
        {
            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException(fileName);
            }

            DataSplit = LoadAndPrepareData(fileName);
            var pipeline = BuildDataProcessingPipeline();
            var trainingPipeline = pipeline.Append(Model);

            TrainedModel = trainingPipeline.Fit(DataSplit.TrainSet);
        }

        public void Save()
        {
            MlContext.Model.Save(TrainedModel, DataSplit.TrainSet.Schema, ModelPath);
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<MovieData>(trainingFileName, hasHeader: true, separatorChar: ',');

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        private EstimatorChain<ValueToKeyMappingTransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "UserId", outputColumnName: "UserIdEncoded")
                                .Append(MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "MovieId", outputColumnName: "MovieIdEncoded"))
                                .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
