using System.IO;
using SentimentAnalysis.MachineLearning.DataModels;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis.MachineLearning.Common
{
    public class TrainerBase<TParamters> : ITrainerBase
        where TParamters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "sentiment.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParamters>, TParamters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(11);
        }

        public BinaryClassificationMetrics Evalute()
        {
            var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);

            return MlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
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
            var trainingDataView = MlContext.Data.LoadFromTextFile<SentimentData>(trainingFileName, hasHeader: false);

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        private EstimatorChain<ITransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "SentimentText")
                                .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
