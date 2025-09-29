# Prompt Versioning & Management System

## Overview

The Prompt Versioning System provides comprehensive management of prompt templates, versioning, A/B testing, and performance tracking. This system ensures prompt quality, enables experimentation, and provides rollback capabilities for production deployments.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt Management System                     │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Prompt Store   │  Version Control│  A/B Testing    │  Analytics│
│  (Database)     │  (Git-like)     │  Engine         │  Engine  │
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  Template       │  Branching      │  Traffic Split  │  Metrics │
│  Management     │  & Merging      │  & Routing      │  Tracking│
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  Validation     │  Diff & Review  │  Statistical    │  Reporting│
│  Engine         │  System         │  Significance   │  & Alerts│
└─────────────────┴─────────────────┴─────────────────┴─────────┘
```

## 1. Prompt Template Management

### Prompt Template Structure
```python
@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata and versioning."""
    
    id: str
    name: str
    description: str
    category: str  # 'qa', 'summarization', 'translation', etc.
    version: str
    content: str
    variables: List[str]  # Template variables like {context}, {question}
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    is_active: bool
    tags: List[str]
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            raise TemplateVariableError(f"Missing required variable: {e}")
```

### Template Management API
```python
class PromptTemplateManager:
    """Manages prompt templates and their lifecycle."""
    
    def __init__(self, storage: PromptStorage, validator: PromptValidator):
        self.storage = storage
        self.validator = validator
        self.version_control = VersionControl()
    
    def create_template(self, template_data: Dict) -> PromptTemplate:
        """Create a new prompt template."""
        # Validate template
        validation_result = self.validator.validate(template_data)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)
        
        # Create template with version
        template = PromptTemplate(
            id=generate_id(),
            version="1.0.0",
            **template_data
        )
        
        # Store in database
        self.storage.save_template(template)
        
        # Initialize version control
        self.version_control.initialize_repository(template.id)
        self.version_control.commit(template.id, template, "Initial version")
        
        return template
    
    def update_template(self, template_id: str, updates: Dict) -> PromptTemplate:
        """Update an existing template with versioning."""
        # Get current template
        current_template = self.storage.get_template(template_id)
        
        # Create new version
        new_version = self._increment_version(current_template.version)
        updated_template = current_template.copy(update=updates, version=new_version)
        
        # Validate updated template
        validation_result = self.validator.validate(updated_template.dict())
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)
        
        # Store new version
        self.storage.save_template(updated_template)
        
        # Commit to version control
        self.version_control.commit(template_id, updated_template, f"Update to version {new_version}")
        
        return updated_template
    
    def get_template(self, template_id: str, version: str = None) -> PromptTemplate:
        """Get a specific template version."""
        if version:
            return self.storage.get_template_version(template_id, version)
        return self.storage.get_latest_template(template_id)
    
    def list_templates(self, category: str = None, tags: List[str] = None) -> List[PromptTemplate]:
        """List templates with optional filtering."""
        return self.storage.list_templates(category=category, tags=tags)
```

## 2. Version Control System

### Git-like Version Control
```python
class VersionControl:
    """Git-like version control for prompt templates."""
    
    def __init__(self, storage: VersionStorage):
        self.storage = storage
        self.branches = {}
        self.commits = {}
    
    def initialize_repository(self, template_id: str):
        """Initialize version control for a template."""
        self.branches[template_id] = {
            'main': Branch(name='main', head_commit=None)
        }
        self.commits[template_id] = []
    
    def commit(self, template_id: str, template: PromptTemplate, message: str) -> str:
        """Create a new commit."""
        commit_id = generate_commit_id()
        commit = Commit(
            id=commit_id,
            template_id=template_id,
            template_version=template.version,
            content=template.content,
            message=message,
            timestamp=datetime.now(),
            author=template.created_by
        )
        
        self.commits[template_id].append(commit)
        
        # Update branch head
        branch = self.branches[template_id]['main']
        branch.head_commit = commit_id
        
        return commit_id
    
    def create_branch(self, template_id: str, branch_name: str, from_commit: str = None) -> str:
        """Create a new branch for experimentation."""
        if from_commit is None:
            from_commit = self.branches[template_id]['main'].head_commit
        
        self.branches[template_id][branch_name] = Branch(
            name=branch_name,
            head_commit=from_commit
        )
        
        return branch_name
    
    def merge_branch(self, template_id: str, source_branch: str, target_branch: str = 'main') -> str:
        """Merge a branch into target branch."""
        source_commit = self.branches[template_id][source_branch].head_commit
        target_commit = self.branches[template_id][target_branch].head_commit
        
        # Create merge commit
        merge_commit = self._create_merge_commit(
            template_id, source_commit, target_commit, 
            f"Merge {source_branch} into {target_branch}"
        )
        
        # Update target branch
        self.branches[template_id][target_branch].head_commit = merge_commit.id
        
        return merge_commit.id
    
    def get_diff(self, template_id: str, commit1: str, commit2: str) -> Diff:
        """Get diff between two commits."""
        commit1_data = self._get_commit(template_id, commit1)
        commit2_data = self._get_commit(template_id, commit2)
        
        return Diff(
            template_id=template_id,
            from_commit=commit1,
            to_commit=commit2,
            changes=self._calculate_changes(commit1_data.content, commit2_data.content)
        )
```

### Branch Management
```python
class Branch:
    """Represents a branch in version control."""
    
    def __init__(self, name: str, head_commit: str = None):
        self.name = name
        self.head_commit = head_commit
        self.created_at = datetime.now()
    
    def get_commit_history(self, version_control: VersionControl, limit: int = 10) -> List[Commit]:
        """Get commit history for this branch."""
        return version_control.get_commit_history(self.head_commit, limit)

class Commit:
    """Represents a single commit."""
    
    def __init__(self, id: str, template_id: str, template_version: str, 
                 content: str, message: str, timestamp: datetime, author: str):
        self.id = id
        self.template_id = template_id
        self.template_version = template_version
        self.content = content
        self.message = message
        self.timestamp = timestamp
        self.author = author
        self.parent_commits = []
```

## 3. A/B Testing Engine

### Experiment Management
```python
class ABTestingEngine:
    """Manages A/B testing for prompt templates."""
    
    def __init__(self, traffic_router: TrafficRouter, metrics_collector: MetricsCollector):
        self.traffic_router = traffic_router
        self.metrics_collector = metrics_collector
        self.experiments = {}
        self.active_experiments = {}
    
    def create_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        experiment_id = generate_experiment_id()
        
        experiment = Experiment(
            id=experiment_id,
            name=experiment_config.name,
            description=experiment_config.description,
            template_id=experiment_config.template_id,
            variants=experiment_config.variants,
            traffic_split=experiment_config.traffic_split,
            start_date=experiment_config.start_date,
            end_date=experiment_config.end_date,
            success_metrics=experiment_config.success_metrics,
            status='draft'
        )
        
        self.experiments[experiment_id] = experiment
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
        
        # Validate experiment configuration
        validation_result = self._validate_experiment(experiment)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)
        
        # Start traffic routing
        self.traffic_router.start_routing(experiment)
        
        # Update experiment status
        experiment.status = 'running'
        experiment.actual_start_date = datetime.now()
        
        self.active_experiments[experiment_id] = experiment
        
        return True
    
    def stop_experiment(self, experiment_id: str) -> ExperimentResults:
        """Stop an A/B test experiment and return results."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(f"Active experiment {experiment_id} not found")
        
        # Stop traffic routing
        self.traffic_router.stop_routing(experiment_id)
        
        # Collect final metrics
        results = self._collect_experiment_results(experiment)
        
        # Update experiment status
        experiment.status = 'completed'
        experiment.actual_end_date = datetime.now()
        experiment.results = results
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        return results
    
    def get_experiment_results(self, experiment_id: str) -> ExperimentResults:
        """Get current results for an active experiment."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(f"Active experiment {experiment_id} not found")
        
        return self._collect_experiment_results(experiment)
```

### Traffic Routing
```python
class TrafficRouter:
    """Routes traffic to different prompt variants in A/B tests."""
    
    def __init__(self, routing_strategy: RoutingStrategy = None):
        self.routing_strategy = routing_strategy or ConsistentHashingStrategy()
        self.active_routes = {}
    
    def start_routing(self, experiment: Experiment):
        """Start routing traffic for an experiment."""
        route_config = RouteConfig(
            experiment_id=experiment.id,
            variants=experiment.variants,
            traffic_split=experiment.traffic_split,
            routing_strategy=self.routing_strategy
        )
        
        self.active_routes[experiment.id] = route_config
    
    def route_request(self, experiment_id: str, user_id: str, request_context: Dict) -> str:
        """Route a request to a specific variant."""
        route_config = self.active_routes.get(experiment_id)
        if not route_config:
            raise ExperimentNotFoundError(f"No active routing for experiment {experiment_id}")
        
        # Determine variant based on routing strategy
        variant = self.routing_strategy.select_variant(
            user_id, request_context, route_config.variants, route_config.traffic_split
        )
        
        return variant
    
    def stop_routing(self, experiment_id: str):
        """Stop routing traffic for an experiment."""
        if experiment_id in self.active_routes:
            del self.active_routes[experiment_id]
```

### Statistical Analysis
```python
class StatisticalAnalyzer:
    """Performs statistical analysis on A/B test results."""
    
    def analyze_experiment(self, experiment: Experiment, metrics_data: List[MetricsData]) -> ExperimentResults:
        """Analyze A/B test results with statistical significance testing."""
        results = ExperimentResults(experiment_id=experiment.id)
        
        # Group metrics by variant
        variant_metrics = self._group_metrics_by_variant(metrics_data)
        
        # Calculate basic statistics for each variant
        for variant_id, metrics in variant_metrics.items():
            variant_stats = self._calculate_variant_statistics(metrics)
            results.variant_results[variant_id] = variant_stats
        
        # Perform statistical significance tests
        if len(variant_metrics) >= 2:
            significance_tests = self._perform_significance_tests(variant_metrics)
            results.significance_tests = significance_tests
        
        # Determine winning variant
        results.winning_variant = self._determine_winning_variant(results.variant_results)
        
        # Calculate confidence intervals
        results.confidence_intervals = self._calculate_confidence_intervals(variant_metrics)
        
        return results
    
    def _perform_significance_tests(self, variant_metrics: Dict[str, List[MetricsData]]) -> Dict[str, Any]:
        """Perform statistical significance tests between variants."""
        from scipy import stats
        
        tests = {}
        variant_ids = list(variant_metrics.keys())
        
        # Compare each pair of variants
        for i in range(len(variant_ids)):
            for j in range(i + 1, len(variant_ids)):
                variant1_id, variant2_id = variant_ids[i], variant_ids[j]
                
                # Extract metrics for comparison
                metrics1 = [m.value for m in variant_metrics[variant1_id]]
                metrics2 = [m.value for m in variant_metrics[variant2_id]]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(metrics1, metrics2)
                
                tests[f"{variant1_id}_vs_{variant2_id}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': self._calculate_effect_size(metrics1, metrics2)
                }
        
        return tests
```

## 4. Performance Tracking & Analytics

### Metrics Collection
```python
class PromptMetricsCollector:
    """Collects and tracks performance metrics for prompt templates."""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        self.metrics_buffer = []
    
    def track_prompt_usage(self, template_id: str, variant_id: str, 
                          request_data: Dict, response_data: Dict, 
                          performance_metrics: PerformanceMetrics):
        """Track usage and performance of a prompt template."""
        metrics = PromptMetrics(
            template_id=template_id,
            variant_id=variant_id,
            timestamp=datetime.now(),
            request_data=request_data,
            response_data=response_data,
            performance_metrics=performance_metrics
        )
        
        self.metrics_buffer.append(metrics)
        
        # Batch write to storage
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def get_template_performance(self, template_id: str, 
                               time_window: str = '24h') -> TemplatePerformance:
        """Get performance metrics for a template."""
        metrics = self.storage.query_metrics(template_id, time_window)
        
        return TemplatePerformance(
            template_id=template_id,
            total_requests=len(metrics),
            average_latency=self._calculate_average_latency(metrics),
            success_rate=self._calculate_success_rate(metrics),
            quality_score=self._calculate_quality_score(metrics),
            cost_per_request=self._calculate_average_cost(metrics),
            error_rate=self._calculate_error_rate(metrics)
        )
```

### Performance Analytics
```python
class PerformanceAnalytics:
    """Analyzes prompt performance and provides insights."""
    
    def __init__(self, metrics_collector: PromptMetricsCollector):
        self.metrics_collector = metrics_collector
    
    def analyze_template_trends(self, template_id: str, 
                               time_period: str = '7d') -> TrendAnalysis:
        """Analyze performance trends for a template."""
        metrics = self.metrics_collector.get_metrics(template_id, time_period)
        
        # Group metrics by time periods
        time_series = self._group_metrics_by_time(metrics)
        
        # Calculate trends
        trends = {}
        for metric_name in ['latency', 'quality_score', 'success_rate']:
            values = [period[metric_name] for period in time_series]
            trend = self._calculate_trend(values)
            trends[metric_name] = trend
        
        return TrendAnalysis(
            template_id=template_id,
            time_period=time_period,
            trends=trends,
            recommendations=self._generate_recommendations(trends)
        )
    
    def compare_variants(self, template_id: str, 
                        variant_ids: List[str]) -> VariantComparison:
        """Compare performance of different variants."""
        variant_performances = {}
        
        for variant_id in variant_ids:
            performance = self.metrics_collector.get_variant_performance(
                template_id, variant_id
            )
            variant_performances[variant_id] = performance
        
        # Calculate statistical significance
        significance_tests = self._perform_variant_comparison(variant_performances)
        
        return VariantComparison(
            template_id=template_id,
            variants=variant_performances,
            significance_tests=significance_tests,
            recommendations=self._generate_variant_recommendations(variant_performances)
        )
```

## 5. Rollback & Deployment Management

### Deployment Pipeline
```python
class PromptDeploymentManager:
    """Manages deployment and rollback of prompt templates."""
    
    def __init__(self, template_manager: PromptTemplateManager, 
                 version_control: VersionControl):
        self.template_manager = template_manager
        self.version_control = version_control
        self.deployment_history = {}
    
    def deploy_template(self, template_id: str, version: str, 
                       environment: str = 'production') -> DeploymentResult:
        """Deploy a specific version of a template."""
        # Get template version
        template = self.template_manager.get_template(template_id, version)
        
        # Validate template
        validation_result = self._validate_for_deployment(template)
        if not validation_result.is_valid:
            return DeploymentResult(
                success=False,
                error=f"Validation failed: {validation_result.errors}"
            )
        
        # Deploy to environment
        deployment_id = self._deploy_to_environment(template, environment)
        
        # Update deployment history
        self.deployment_history[deployment_id] = DeploymentRecord(
            deployment_id=deployment_id,
            template_id=template_id,
            version=version,
            environment=environment,
            deployed_at=datetime.now(),
            status='deployed'
        )
        
        return DeploymentResult(
            success=True,
            deployment_id=deployment_id,
            template_version=version
        )
    
    def rollback_template(self, template_id: str, 
                         target_version: str = None) -> RollbackResult:
        """Rollback template to a previous version."""
        if target_version is None:
            # Rollback to previous version
            target_version = self._get_previous_version(template_id)
        
        # Deploy target version
        deployment_result = self.deploy_template(template_id, target_version)
        
        if deployment_result.success:
            return RollbackResult(
                success=True,
                rolled_back_to=target_version,
                deployment_id=deployment_result.deployment_id
            )
        else:
            return RollbackResult(
                success=False,
                error=deployment_result.error
            )
    
    def get_deployment_history(self, template_id: str) -> List[DeploymentRecord]:
        """Get deployment history for a template."""
        return [record for record in self.deployment_history.values() 
                if record.template_id == template_id]
```

## 6. Configuration & Management

### System Configuration
```yaml
# prompt_versioning.yaml
prompt_versioning:
  storage:
    type: postgresql
    host: localhost
    port: 5432
    database: prompt_templates
    username: prompt_user
    password: ${PROMPT_DB_PASSWORD}
  
  version_control:
    storage_type: git
    repository_path: /var/lib/prompt-versioning
    branch_strategy: feature_branches
  
  ab_testing:
    traffic_routing:
      strategy: consistent_hashing
      hash_function: md5
    
    statistical_analysis:
      significance_level: 0.05
      minimum_sample_size: 100
      confidence_interval: 0.95
  
  performance_tracking:
    metrics_retention_days: 90
    batch_size: 100
    flush_interval: 60  # seconds
  
  deployment:
    environments:
      - development
      - staging
      - production
    
    validation:
      required_checks:
        - syntax_validation
        - variable_validation
        - performance_validation
        - security_validation
```

### API Endpoints
```python
# FastAPI endpoints for prompt management
@app.post("/api/v1/prompts")
async def create_prompt_template(template_data: PromptTemplateCreate):
    """Create a new prompt template."""
    return await prompt_manager.create_template(template_data)

@app.get("/api/v1/prompts/{template_id}")
async def get_prompt_template(template_id: str, version: str = None):
    """Get a prompt template."""
    return await prompt_manager.get_template(template_id, version)

@app.put("/api/v1/prompts/{template_id}")
async def update_prompt_template(template_id: str, updates: PromptTemplateUpdate):
    """Update a prompt template."""
    return await prompt_manager.update_template(template_id, updates)

@app.post("/api/v1/prompts/{template_id}/experiments")
async def create_experiment(template_id: str, experiment_config: ExperimentConfig):
    """Create an A/B test experiment."""
    return await ab_testing_engine.create_experiment(experiment_config)

@app.post("/api/v1/prompts/{template_id}/deploy")
async def deploy_template(template_id: str, version: str, environment: str = "production"):
    """Deploy a template version."""
    return await deployment_manager.deploy_template(template_id, version, environment)

@app.post("/api/v1/prompts/{template_id}/rollback")
async def rollback_template(template_id: str, target_version: str = None):
    """Rollback to a previous version."""
    return await deployment_manager.rollback_template(template_id, target_version)
```

## 7. Monitoring & Alerting

### Performance Monitoring
```python
class PromptMonitoring:
    """Monitors prompt performance and triggers alerts."""
    
    def __init__(self, metrics_collector: PromptMetricsCollector, 
                 alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.alert_rules = {}
    
    def setup_alert_rule(self, rule: AlertRule):
        """Setup an alert rule for prompt monitoring."""
        self.alert_rules[rule.id] = rule
    
    def check_alerts(self, template_id: str):
        """Check if any alerts should be triggered."""
        performance = self.metrics_collector.get_template_performance(template_id)
        
        for rule in self.alert_rules.values():
            if rule.template_id == template_id or rule.template_id == "*":
                if self._evaluate_rule(rule, performance):
                    self.alert_manager.send_alert(rule, performance)
    
    def _evaluate_rule(self, rule: AlertRule, performance: TemplatePerformance) -> bool:
        """Evaluate if an alert rule should trigger."""
        if rule.metric == 'latency' and performance.average_latency > rule.threshold:
            return True
        elif rule.metric == 'error_rate' and performance.error_rate > rule.threshold:
            return True
        elif rule.metric == 'quality_score' and performance.quality_score < rule.threshold:
            return True
        
        return False
```

This comprehensive prompt versioning system provides enterprise-grade management of prompt templates with full version control, A/B testing, performance tracking, and deployment management capabilities.
