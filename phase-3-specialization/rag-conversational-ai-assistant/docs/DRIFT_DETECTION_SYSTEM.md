# Drift Detection & Monitoring System

## Overview

The Drift Detection & Monitoring System provides comprehensive monitoring and detection of various types of drift in the RAG Conversational AI Assistant. This system ensures model performance remains optimal over time by detecting changes in data distribution, model performance, and user behavior patterns.

## Types of Drift

### 1. Data Drift
- **Input Data Distribution**: Changes in the distribution of input queries
- **Feature Drift**: Changes in the statistical properties of input features
- **Concept Drift**: Changes in the relationship between inputs and outputs
- **Covariate Shift**: Changes in the distribution of input variables

### 2. Model Drift
- **Performance Degradation**: Decline in model accuracy over time
- **Prediction Drift**: Changes in model predictions for similar inputs
- **Confidence Drift**: Changes in model confidence scores
- **Latency Drift**: Changes in model response times

### 3. Concept Drift
- **Query Pattern Changes**: Evolution in user query patterns and topics
- **User Behavior Changes**: Changes in how users interact with the system
- **Domain Evolution**: Changes in the knowledge domain being queried
- **Seasonal Patterns**: Time-based changes in query patterns

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                Drift Detection & Monitoring System             │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Data Drift     │  Model Drift    │  Concept Drift  │  Alert  │
│  Detection      │  Detection      │  Detection      │  System │
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  Statistical    │  Performance    │  Pattern        │  Real-time│
│  Tests          │  Monitoring     │  Analysis       │  Alerts │
│  Distribution   │  Accuracy       │  User Behavior  │  &      │
│  Analysis       │  Tracking       │  Analysis       │  Notifications│
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response & Mitigation Layer                 │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Automated      │  Model          │  Data           │  Human  │
│  Retraining     │  Rollback       │  Pipeline       │  Review │
│  Triggers       │  Mechanisms     │  Updates        │  Workflow│
└─────────────────┴─────────────────┴─────────────────┴─────────┘
```

## 1. Data Drift Detection

### Statistical Drift Detection
```python
class DataDriftDetector:
    """Detects data drift using statistical methods."""
    
    def __init__(self, reference_data: np.ndarray, drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.drift_tests = {
            'ks_test': KolmogorovSmirnovTest(),
            'wasserstein': WassersteinDistance(),
            'kl_divergence': KLDivergenceTest(),
            'psi': PopulationStabilityIndex()
        }
    
    async def detect_drift(self, current_data: np.ndarray) -> DriftResult:
        """Detect drift in current data compared to reference data."""
        drift_results = {}
        
        for test_name, test in self.drift_tests.items():
            try:
                result = await test.calculate_drift(self.reference_data, current_data)
                drift_results[test_name] = result
            except Exception as e:
                logger.error(f"Error in {test_name} drift test: {e}")
                drift_results[test_name] = None
        
        # Determine overall drift status
        overall_drift = self._determine_overall_drift(drift_results)
        
        return DriftResult(
            timestamp=datetime.now(),
            drift_detected=overall_drift,
            individual_tests=drift_results,
            severity=self._calculate_severity(drift_results),
            recommendations=self._generate_recommendations(drift_results)
        )
    
    def _determine_overall_drift(self, drift_results: Dict) -> bool:
        """Determine if drift is detected based on individual test results."""
        significant_tests = 0
        total_tests = 0
        
        for test_name, result in drift_results.items():
            if result is not None:
                total_tests += 1
                if result.p_value < self.drift_threshold:
                    significant_tests += 1
        
        # Drift detected if majority of tests show significant drift
        return significant_tests > total_tests / 2
    
    def _calculate_severity(self, drift_results: Dict) -> str:
        """Calculate drift severity level."""
        significant_count = sum(1 for r in drift_results.values() 
                               if r and r.p_value < self.drift_threshold)
        
        if significant_count >= 3:
            return 'critical'
        elif significant_count >= 2:
            return 'high'
        elif significant_count >= 1:
            return 'medium'
        else:
            return 'low'

class KolmogorovSmirnovTest:
    """Kolmogorov-Smirnov test for distribution drift."""
    
    async def calculate_drift(self, reference_data: np.ndarray, 
                            current_data: np.ndarray) -> DriftTestResult:
        """Calculate KS test drift between reference and current data."""
        from scipy.stats import ks_2samp
        
        statistic, p_value = ks_2samp(reference_data, current_data)
        
        return DriftTestResult(
            test_name='ks_test',
            statistic=statistic,
            p_value=p_value,
            drift_detected=p_value < 0.05,
            confidence_level=1 - p_value
        )

class PopulationStabilityIndex:
    """Population Stability Index for drift detection."""
    
    def __init__(self, bins: int = 10):
        self.bins = bins
    
    async def calculate_drift(self, reference_data: np.ndarray, 
                            current_data: np.ndarray) -> DriftTestResult:
        """Calculate PSI between reference and current data."""
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference_data, bins=self.bins)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference_data, bins=bin_edges)
        curr_hist, _ = np.histogram(current_data, bins=bin_edges)
        
        # Normalize to probabilities
        ref_probs = ref_hist / len(reference_data)
        curr_probs = curr_hist / len(current_data)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        # PSI interpretation: < 0.1 stable, 0.1-0.2 moderate change, > 0.2 significant change
        drift_detected = psi > 0.1
        
        return DriftTestResult(
            test_name='psi',
            statistic=psi,
            p_value=1 - psi,  # Approximate p-value
            drift_detected=drift_detected,
            confidence_level=min(1.0, psi * 5)  # Approximate confidence
        )
```

### Feature Drift Detection
```python
class FeatureDriftDetector:
    """Detects drift in specific features of the input data."""
    
    def __init__(self, feature_configs: Dict[str, FeatureConfig]):
        self.feature_configs = feature_configs
        self.drift_detectors = {}
        
        for feature_name, config in feature_configs.items():
            self.drift_detectors[feature_name] = self._create_detector(config)
    
    def _create_detector(self, config: FeatureConfig) -> BaseDriftDetector:
        """Create appropriate drift detector for feature type."""
        if config.feature_type == 'numerical':
            return NumericalDriftDetector(config)
        elif config.feature_type == 'categorical':
            return CategoricalDriftDetector(config)
        elif config.feature_type == 'text':
            return TextDriftDetector(config)
        else:
            raise ValueError(f"Unsupported feature type: {config.feature_type}")
    
    async def detect_feature_drift(self, reference_data: Dict[str, np.ndarray], 
                                 current_data: Dict[str, np.ndarray]) -> Dict[str, DriftResult]:
        """Detect drift for each feature."""
        feature_drift_results = {}
        
        for feature_name in self.feature_configs.keys():
            if feature_name in reference_data and feature_name in current_data:
                detector = self.drift_detectors[feature_name]
                result = await detector.detect_drift(
                    reference_data[feature_name],
                    current_data[feature_name]
                )
                feature_drift_results[feature_name] = result
        
        return feature_drift_results

class TextDriftDetector:
    """Detects drift in text features using embedding-based methods."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.drift_threshold = config.drift_threshold
    
    async def detect_drift(self, reference_texts: List[str], 
                          current_texts: List[str]) -> DriftResult:
        """Detect drift in text data using embedding similarity."""
        # Generate embeddings
        ref_embeddings = await self._generate_embeddings(reference_texts)
        curr_embeddings = await self._generate_embeddings(current_texts)
        
        # Calculate distribution similarity
        similarity = self._calculate_embedding_similarity(ref_embeddings, curr_embeddings)
        
        # Determine drift
        drift_detected = similarity < self.drift_threshold
        
        return DriftResult(
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            similarity_score=similarity,
            severity='high' if drift_detected else 'low',
            recommendations=self._generate_text_drift_recommendations(similarity)
        )
    
    def _calculate_embedding_similarity(self, ref_embeddings: np.ndarray, 
                                      curr_embeddings: np.ndarray) -> float:
        """Calculate similarity between embedding distributions."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate mean embeddings
        ref_mean = np.mean(ref_embeddings, axis=0)
        curr_mean = np.mean(curr_embeddings, axis=0)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([ref_mean], [curr_mean])[0][0]
        
        return similarity
```

## 2. Model Drift Detection

### Performance Drift Detection
```python
class ModelDriftDetector:
    """Detects drift in model performance over time."""
    
    def __init__(self, performance_thresholds: Dict[str, float]):
        self.performance_thresholds = performance_thresholds
        self.performance_tracker = PerformanceTracker()
        self.drift_analyzer = ModelDriftAnalyzer()
    
    async def detect_model_drift(self, time_window: str = '7d') -> ModelDriftResult:
        """Detect model performance drift over time."""
        # Get performance metrics for the time window
        performance_data = await self.performance_tracker.get_performance_data(time_window)
        
        # Analyze performance trends
        trend_analysis = await self.drift_analyzer.analyze_performance_trends(performance_data)
        
        # Check for significant performance degradation
        drift_detected = self._check_performance_drift(trend_analysis)
        
        # Calculate drift severity
        severity = self._calculate_drift_severity(trend_analysis)
        
        return ModelDriftResult(
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            performance_trends=trend_analysis,
            severity=severity,
            recommendations=self._generate_model_drift_recommendations(trend_analysis)
        )
    
    def _check_performance_drift(self, trend_analysis: Dict) -> bool:
        """Check if performance drift is detected."""
        # Check accuracy drift
        accuracy_drift = trend_analysis['accuracy']['slope'] < -0.01  # 1% decrease per day
        
        # Check latency drift
        latency_drift = trend_analysis['latency']['slope'] > 0.1  # 0.1s increase per day
        
        # Check error rate drift
        error_rate_drift = trend_analysis['error_rate']['slope'] > 0.001  # 0.1% increase per day
        
        return accuracy_drift or latency_drift or error_rate_drift
    
    def _calculate_drift_severity(self, trend_analysis: Dict) -> str:
        """Calculate the severity of model drift."""
        severity_scores = []
        
        # Accuracy severity
        accuracy_slope = trend_analysis['accuracy']['slope']
        if accuracy_slope < -0.05:
            severity_scores.append('critical')
        elif accuracy_slope < -0.02:
            severity_scores.append('high')
        elif accuracy_slope < -0.01:
            severity_scores.append('medium')
        else:
            severity_scores.append('low')
        
        # Latency severity
        latency_slope = trend_analysis['latency']['slope']
        if latency_slope > 0.5:
            severity_scores.append('critical')
        elif latency_slope > 0.2:
            severity_scores.append('high')
        elif latency_slope > 0.1:
            severity_scores.append('medium')
        else:
            severity_scores.append('low')
        
        # Return highest severity
        severity_order = ['low', 'medium', 'high', 'critical']
        return max(severity_scores, key=lambda x: severity_order.index(x))

class ModelDriftAnalyzer:
    """Analyzes model performance trends and drift patterns."""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.statistical_tests = StatisticalTests()
    
    async def analyze_performance_trends(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Extract time series data
        timestamps = [d['timestamp'] for d in performance_data]
        accuracy_scores = [d['accuracy'] for d in performance_data]
        latency_scores = [d['latency'] for d in performance_data]
        error_rates = [d['error_rate'] for d in performance_data]
        
        # Calculate trends
        accuracy_trend = self.trend_analyzer.calculate_trend(accuracy_scores)
        latency_trend = self.trend_analyzer.calculate_trend(latency_scores)
        error_rate_trend = self.trend_analyzer.calculate_trend(error_rates)
        
        # Perform statistical tests
        accuracy_significance = self.statistical_tests.mann_kendall_test(accuracy_scores)
        latency_significance = self.statistical_tests.mann_kendall_test(latency_scores)
        error_rate_significance = self.statistical_tests.mann_kendall_test(error_rates)
        
        return {
            'accuracy': {
                'values': accuracy_scores,
                'slope': accuracy_trend['slope'],
                'significance': accuracy_significance,
                'trend': accuracy_trend['trend']
            },
            'latency': {
                'values': latency_scores,
                'slope': latency_trend['slope'],
                'significance': latency_significance,
                'trend': latency_trend['trend']
            },
            'error_rate': {
                'values': error_rates,
                'slope': error_rate_trend['slope'],
                'significance': error_rate_significance,
                'trend': error_rate_trend['trend']
            }
        }
```

### Prediction Drift Detection
```python
class PredictionDriftDetector:
    """Detects drift in model predictions over time."""
    
    def __init__(self, reference_predictions: List[Dict], drift_threshold: float = 0.1):
        self.reference_predictions = reference_predictions
        self.drift_threshold = drift_threshold
        self.prediction_analyzer = PredictionAnalyzer()
    
    async def detect_prediction_drift(self, current_predictions: List[Dict]) -> PredictionDriftResult:
        """Detect drift in current predictions compared to reference."""
        # Extract prediction features
        ref_features = self._extract_prediction_features(self.reference_predictions)
        curr_features = self._extract_prediction_features(current_predictions)
        
        # Calculate drift metrics
        drift_metrics = await self.prediction_analyzer.calculate_drift_metrics(
            ref_features, curr_features
        )
        
        # Determine drift status
        drift_detected = self._evaluate_drift_metrics(drift_metrics)
        
        return PredictionDriftResult(
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_metrics=drift_metrics,
            severity=self._calculate_prediction_drift_severity(drift_metrics),
            recommendations=self._generate_prediction_drift_recommendations(drift_metrics)
        )
    
    def _extract_prediction_features(self, predictions: List[Dict]) -> Dict[str, List]:
        """Extract features from predictions for drift analysis."""
        features = {
            'confidence_scores': [p['confidence'] for p in predictions],
            'response_lengths': [len(p['response']) for p in predictions],
            'source_counts': [len(p['sources']) for p in predictions],
            'response_types': [p['response_type'] for p in predictions]
        }
        return features
    
    def _evaluate_drift_metrics(self, drift_metrics: Dict) -> bool:
        """Evaluate if drift is detected based on metrics."""
        # Check if any metric exceeds threshold
        for metric_name, value in drift_metrics.items():
            if value > self.drift_threshold:
                return True
        return False
```

## 3. Concept Drift Detection

### Query Pattern Drift Detection
```python
class ConceptDriftDetector:
    """Detects concept drift in user queries and behavior patterns."""
    
    def __init__(self, reference_period: str = '30d'):
        self.reference_period = reference_period
        self.pattern_analyzer = PatternAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
    
    async def detect_concept_drift(self, current_period: str = '7d') -> ConceptDriftResult:
        """Detect concept drift in the current period."""
        # Get reference and current data
        reference_data = await self._get_reference_data(self.reference_period)
        current_data = await self._get_current_data(current_period)
        
        # Analyze different aspects of concept drift
        topic_drift = await self._analyze_topic_drift(reference_data, current_data)
        pattern_drift = await self._analyze_pattern_drift(reference_data, current_data)
        behavior_drift = await self._analyze_behavior_drift(reference_data, current_data)
        
        # Determine overall concept drift
        overall_drift = self._determine_overall_concept_drift(
            topic_drift, pattern_drift, behavior_drift
        )
        
        return ConceptDriftResult(
            timestamp=datetime.now(),
            drift_detected=overall_drift,
            topic_drift=topic_drift,
            pattern_drift=pattern_drift,
            behavior_drift=behavior_drift,
            severity=self._calculate_concept_drift_severity(
                topic_drift, pattern_drift, behavior_drift
            ),
            recommendations=self._generate_concept_drift_recommendations(
                topic_drift, pattern_drift, behavior_drift
            )
        )
    
    async def _analyze_topic_drift(self, reference_data: Dict, current_data: Dict) -> TopicDriftResult:
        """Analyze drift in query topics."""
        # Extract topics from queries
        ref_topics = await self.topic_analyzer.extract_topics(reference_data['queries'])
        curr_topics = await self.topic_analyzer.extract_topics(current_data['queries'])
        
        # Calculate topic distribution drift
        topic_drift_score = self._calculate_topic_drift_score(ref_topics, curr_topics)
        
        # Identify new topics
        new_topics = self._identify_new_topics(ref_topics, curr_topics)
        
        # Identify disappearing topics
        disappearing_topics = self._identify_disappearing_topics(ref_topics, curr_topics)
        
        return TopicDriftResult(
            drift_score=topic_drift_score,
            new_topics=new_topics,
            disappearing_topics=disappearing_topics,
            drift_detected=topic_drift_score > 0.2
        )
    
    async def _analyze_pattern_drift(self, reference_data: Dict, current_data: Dict) -> PatternDriftResult:
        """Analyze drift in query patterns."""
        # Extract query patterns
        ref_patterns = await self.pattern_analyzer.extract_patterns(reference_data['queries'])
        curr_patterns = await self.pattern_analyzer.extract_patterns(current_data['queries'])
        
        # Calculate pattern similarity
        pattern_similarity = self._calculate_pattern_similarity(ref_patterns, curr_patterns)
        
        # Identify pattern changes
        pattern_changes = self._identify_pattern_changes(ref_patterns, curr_patterns)
        
        return PatternDriftResult(
            similarity_score=pattern_similarity,
            pattern_changes=pattern_changes,
            drift_detected=pattern_similarity < 0.7
        )
    
    async def _analyze_behavior_drift(self, reference_data: Dict, current_data: Dict) -> BehaviorDriftResult:
        """Analyze drift in user behavior patterns."""
        # Extract behavior features
        ref_behavior = await self.behavior_analyzer.extract_behavior_features(reference_data)
        curr_behavior = await self.behavior_analyzer.extract_behavior_features(current_data)
        
        # Calculate behavior drift
        behavior_drift_score = self._calculate_behavior_drift_score(ref_behavior, curr_behavior)
        
        # Identify behavior changes
        behavior_changes = self._identify_behavior_changes(ref_behavior, curr_behavior)
        
        return BehaviorDriftResult(
            drift_score=behavior_drift_score,
            behavior_changes=behavior_changes,
            drift_detected=behavior_drift_score > 0.15
        )
```

## 4. Alert System

### Drift Alert Management
```python
class DriftAlertSystem:
    """Manages drift detection alerts and notifications."""
    
    def __init__(self, alert_rules: List[DriftAlertRule], notification_service: NotificationService):
        self.alert_rules = alert_rules
        self.notification_service = notification_service
        self.active_alerts = {}
        self.alert_history = []
    
    async def process_drift_result(self, drift_result: DriftResult):
        """Process a drift detection result and trigger alerts if needed."""
        for rule in self.alert_rules:
            if await self._evaluate_alert_rule(rule, drift_result):
                await self._trigger_drift_alert(rule, drift_result)
    
    async def _evaluate_alert_rule(self, rule: DriftAlertRule, drift_result: DriftResult) -> bool:
        """Evaluate if an alert rule should trigger for the drift result."""
        if rule.drift_type != drift_result.drift_type:
            return False
        
        if rule.severity_threshold and drift_result.severity != rule.severity_threshold:
            return False
        
        if rule.metric_threshold and drift_result.metric_value < rule.metric_threshold:
            return False
        
        return True
    
    async def _trigger_drift_alert(self, rule: DriftAlertRule, drift_result: DriftResult):
        """Trigger a drift alert."""
        alert_id = f"drift_{rule.name}_{datetime.now().timestamp()}"
        
        alert = DriftAlert(
            id=alert_id,
            rule_name=rule.name,
            drift_type=drift_result.drift_type,
            severity=drift_result.severity,
            message=rule.message_template.format(**drift_result.to_dict()),
            drift_result=drift_result,
            timestamp=datetime.now(),
            status='active'
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notification
        await self.notification_service.send_drift_alert(alert)
        
        # Log alert
        logger.warning(f"Drift alert triggered: {rule.name} - {alert.message}")
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = None):
        """Resolve a drift alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = 'resolved'
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Send resolution notification
            await self.notification_service.send_alert_resolution(alert)
```

### Automated Response System
```python
class DriftResponseSystem:
    """Automated response system for drift detection."""
    
    def __init__(self, model_manager: ModelManager, data_pipeline: DataPipeline):
        self.model_manager = model_manager
        self.data_pipeline = data_pipeline
        self.response_actions = {
            'retrain_model': self._retrain_model,
            'rollback_model': self._rollback_model,
            'update_data_pipeline': self._update_data_pipeline,
            'adjust_thresholds': self._adjust_thresholds,
            'notify_team': self._notify_team
        }
    
    async def handle_drift_alert(self, alert: DriftAlert):
        """Handle a drift alert with automated responses."""
        # Determine appropriate response actions
        response_actions = self._determine_response_actions(alert)
        
        # Execute response actions
        for action in response_actions:
            try:
                await self.response_actions[action](alert)
                logger.info(f"Executed response action: {action} for alert {alert.id}")
            except Exception as e:
                logger.error(f"Error executing response action {action}: {e}")
    
    def _determine_response_actions(self, alert: DriftAlert) -> List[str]:
        """Determine appropriate response actions for an alert."""
        actions = []
        
        if alert.drift_type == 'model_drift' and alert.severity in ['high', 'critical']:
            actions.append('retrain_model')
            actions.append('notify_team')
        
        if alert.drift_type == 'data_drift' and alert.severity == 'critical':
            actions.append('update_data_pipeline')
            actions.append('notify_team')
        
        if alert.drift_type == 'concept_drift':
            actions.append('adjust_thresholds')
            actions.append('notify_team')
        
        return actions
    
    async def _retrain_model(self, alert: DriftAlert):
        """Retrain the model in response to drift."""
        logger.info(f"Retraining model due to drift alert: {alert.id}")
        
        # Trigger model retraining
        await self.model_manager.retrain_model(
            reason=f"Drift alert: {alert.rule_name}",
            alert_id=alert.id
        )
    
    async def _rollback_model(self, alert: DriftAlert):
        """Rollback to previous model version."""
        logger.info(f"Rolling back model due to drift alert: {alert.id}")
        
        # Rollback to previous stable version
        await self.model_manager.rollback_model(
            reason=f"Drift alert: {alert.rule_name}",
            alert_id=alert.id
        )
    
    async def _update_data_pipeline(self, alert: DriftAlert):
        """Update data pipeline in response to drift."""
        logger.info(f"Updating data pipeline due to drift alert: {alert.id}")
        
        # Update data processing pipeline
        await self.data_pipeline.update_pipeline(
            reason=f"Drift alert: {alert.rule_name}",
            alert_id=alert.id
        )
    
    async def _adjust_thresholds(self, alert: DriftAlert):
        """Adjust drift detection thresholds."""
        logger.info(f"Adjusting thresholds due to drift alert: {alert.id}")
        
        # Adjust drift detection thresholds
        await self._update_drift_thresholds(alert)
    
    async def _notify_team(self, alert: DriftAlert):
        """Notify the team about the drift alert."""
        logger.info(f"Notifying team about drift alert: {alert.id}")
        
        # Send team notification
        await self.notification_service.send_team_notification(alert)
```

## 5. Configuration & Management

### Drift Detection Configuration
```yaml
# drift_detection_config.yaml
drift_detection:
  data_drift:
    enabled: true
    detection_interval: 3600  # seconds
    reference_window: "30d"
    current_window: "1d"
    threshold: 0.05
    tests:
      - ks_test
      - wasserstein
      - kl_divergence
      - psi
    
  model_drift:
    enabled: true
    detection_interval: 1800  # seconds
    performance_window: "7d"
    accuracy_threshold: 0.02  # 2% decrease
    latency_threshold: 0.1    # 0.1s increase
    error_rate_threshold: 0.001  # 0.1% increase
    
  concept_drift:
    enabled: true
    detection_interval: 7200  # seconds
    reference_period: "30d"
    current_period: "7d"
    topic_drift_threshold: 0.2
    pattern_similarity_threshold: 0.7
    behavior_drift_threshold: 0.15
  
  alerts:
    rules:
      - name: critical_model_drift
        drift_type: model_drift
        severity_threshold: critical
        actions: [retrain_model, notify_team]
        
      - name: high_data_drift
        drift_type: data_drift
        severity_threshold: high
        actions: [update_data_pipeline, notify_team]
        
      - name: concept_drift_detected
        drift_type: concept_drift
        metric_threshold: 0.1
        actions: [adjust_thresholds, notify_team]
    
    notifications:
      email:
        enabled: true
        recipients: ["admin@company.com", "ml-team@company.com"]
      
      slack:
        enabled: true
        webhook_url: "https://hooks.slack.com/services/..."
        channel: "#ml-alerts"
      
      webhook:
        enabled: true
        url: "https://internal-api.company.com/drift-alerts"
  
  response:
    automated_actions:
      enabled: true
      retrain_on_critical_model_drift: true
      rollback_on_severe_drift: true
      update_pipeline_on_data_drift: true
    
    human_review:
      enabled: true
      escalation_threshold: high
      review_timeout: 3600  # seconds
```

This comprehensive Drift Detection & Monitoring System provides enterprise-grade monitoring and detection of various types of drift in the RAG Conversational AI Assistant, ensuring optimal performance and continuous improvement.
