using RandomForestRegression.MachineLearning.DataModels;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace RandomForestRegression.MachineLearning.Common
{
    public class TrainerBase<TParamters> : ITrainerBase
        where TParamters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "RandomForestRegressionModel.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<RegressionPredictionTransformer<TParamters>, TParamters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(11);
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
            var trainingDataView = MlContext.Data.LoadFromTextFile<HousingData>(trainingFileName, hasHeader: true, separatorChar: ',');

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.CopyColumns("Label", nameof(HousingData.MedianPrice))
                                .Append(MlContext.Transforms.Categorical.OneHotEncoding("RiverCoast"))
                                .Append(MlContext.Transforms.Concatenate("Features",
                                    nameof(HousingData.CrimeRate),
                                    nameof(HousingData.Zoned),
                                    nameof(HousingData.Proportion),
                                    nameof(HousingData.RiverCoast),
                                    nameof(HousingData.NOConcentration),
                                    nameof(HousingData.NumOfRoomsPerDwelling),
                                    nameof(HousingData.Age),
                                    nameof(HousingData.EmployCenterDistance),
                                    nameof(HousingData.HighwayAccessabilityRadius),
                                    nameof(HousingData.TaxRate),
                                    nameof(HousingData.PTRatio),
                                    nameof(HousingData.MedianPrice)))
                                .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
