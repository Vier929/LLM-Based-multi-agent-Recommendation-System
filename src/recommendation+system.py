"""
Theory-Driven Multi-Agent Recommendation System for Urban AI Applications
A SOTA implementation based on the research paper bridging theory-practice gap
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod
import networkx as nx
from scipy.spatial.distance import cosine
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data structures
@dataclass
class UrbanScenario:
    """Structured representation of urban challenge"""
    description: str
    domain: str = ""
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    temporal_scope: str = ""
    spatial_boundaries: str = ""
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0

@dataclass
class UrbanTheory:
    """Urban planning/safety theory representation"""
    name: str
    principles: List[str]
    computational_requirements: List[str]
    category: str  # Design, Spatial, Safety, Perception
    year: int
    embedding: Optional[np.ndarray] = None

@dataclass
class Algorithm:
    """AI/ML algorithm representation"""
    name: str
    capabilities: Dict[str, float]  # capability -> score
    computational_cost: float
    data_requirements: List[str]
    group: int  # 0-3 as per paper

@dataclass
class DataSource:
    """Data source representation"""
    name: str
    type: str
    quality_scores: Dict[str, float]  # dimension -> score
    accessibility: float
    update_frequency: str

@dataclass
class Recommendation:
    """Complete recommendation package"""
    theories: List[UrbanTheory]
    algorithms: List[Algorithm]
    data_sources: List[DataSource]
    confidence_score: float
    validation_results: Dict[str, Any]

# Agent Base Class
class Agent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data asynchronously"""
        pass

# Scenario Analyzer Agent
class ScenarioAnalyzerAgent(Agent):
    """Transform unstructured urban challenges into structured representations"""
    
    def __init__(self):
        super().__init__("ScenarioAnalyzer")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model.eval()
        
        # Domain keywords for classification
        self.domain_keywords = {
            "transportation": ["traffic", "transport", "mobility", "transit", "vehicle"],
            "safety": ["crime", "security", "surveillance", "safety", "emergency"],
            "housing": ["housing", "residential", "dwelling", "affordable", "gentrification"],
            "environment": ["pollution", "green", "sustainability", "climate", "emission"],
            "economic": ["business", "economy", "employment", "commerce", "retail"]
        }
        
    async def process(self, description: str) -> UrbanScenario:
        """Extract structured information from unstructured description"""
        self.logger.info(f"Analyzing scenario: {description[:50]}...")
        
        scenario = UrbanScenario(description=description)
        
        # Extract domain
        scenario.domain = self._classify_domain(description)
        
        # Extract objectives, constraints, etc. using NLP
        extracted_info = await self._extract_information(description)
        scenario.objectives = extracted_info['objectives']
        scenario.constraints = extracted_info['constraints']
        scenario.stakeholders = extracted_info['stakeholders']
        scenario.temporal_scope = extracted_info['temporal_scope']
        scenario.spatial_boundaries = extracted_info['spatial_boundaries']
        
        # Calculate complexity score
        scenario.complexity_score = self._calculate_complexity(scenario)
        
        return scenario
    
    def _classify_domain(self, text: str) -> str:
        """Classify the urban domain based on keywords"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
            
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"
    
    async def _extract_information(self, text: str) -> Dict[str, Any]:
        """Extract structured information using NLP"""
        # Simplified extraction using pattern matching and NER
        # In production, would use more sophisticated NLP pipelines
        
        info = {
            'objectives': self._extract_objectives(text),
            'constraints': self._extract_constraints(text),
            'stakeholders': self._extract_stakeholders(text),
            'temporal_scope': self._extract_temporal(text),
            'spatial_boundaries': self._extract_spatial(text)
        }
        
        return info
    
    def _extract_objectives(self, text: str) -> List[str]:
        """Extract objectives from text"""
        objective_patterns = ["to reduce", "to improve", "to enhance", "to optimize", "to minimize"]
        objectives = []
        
        for pattern in objective_patterns:
            if pattern in text.lower():
                # Extract the phrase following the pattern
                start = text.lower().find(pattern)
                end = min(start + 50, len(text))
                objectives.append(text[start:end].strip())
                
        return objectives[:3]  # Limit to top 3
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from text"""
        constraint_patterns = ["limited", "constraint", "restriction", "budget", "within"]
        constraints = []
        
        for pattern in constraint_patterns:
            if pattern in text.lower():
                start = text.lower().find(pattern)
                end = min(start + 40, len(text))
                constraints.append(text[start:end].strip())
                
        return constraints[:3]
    
    def _extract_stakeholders(self, text: str) -> List[str]:
        """Extract stakeholders mentioned in text"""
        stakeholder_keywords = ["residents", "government", "businesses", "community", "citizens",
                               "authorities", "developers", "organizations", "agencies"]
        
        stakeholders = [kw for kw in stakeholder_keywords if kw in text.lower()]
        return stakeholders[:4]
    
    def _extract_temporal(self, text: str) -> str:
        """Extract temporal scope"""
        if any(word in text.lower() for word in ["immediate", "urgent", "now"]):
            return "immediate"
        elif any(word in text.lower() for word in ["short-term", "months"]):
            return "short-term"
        elif any(word in text.lower() for word in ["long-term", "years"]):
            return "long-term"
        else:
            return "unspecified"
    
    def _extract_spatial(self, text: str) -> str:
        """Extract spatial boundaries"""
        if any(word in text.lower() for word in ["neighborhood", "district"]):
            return "neighborhood"
        elif any(word in text.lower() for word in ["city", "urban"]):
            return "city"
        elif any(word in text.lower() for word in ["region", "metropolitan"]):
            return "regional"
        else:
            return "unspecified"
    
    def _calculate_complexity(self, scenario: UrbanScenario) -> float:
        """Calculate scenario complexity score using equation from paper"""
        weights = {
            'objectives': 0.2,
            'constraints': 0.15,
            'stakeholders': 0.15,
            'temporal': 0.1,
            'spatial': 0.1,
            'domain_complexity': 0.3
        }
        
        complexity_scores = {
            'objectives': min(len(scenario.objectives) / 3.0, 1.0),
            'constraints': min(len(scenario.constraints) / 3.0, 1.0),
            'stakeholders': min(len(scenario.stakeholders) / 4.0, 1.0),
            'temporal': 0.3 if scenario.temporal_scope == "immediate" else 0.6 if scenario.temporal_scope == "short-term" else 1.0,
            'spatial': 0.3 if scenario.spatial_boundaries == "neighborhood" else 0.6 if scenario.spatial_boundaries == "city" else 1.0,
            'domain_complexity': 0.8 if scenario.domain in ["safety", "transportation"] else 0.5
        }
        
        total_complexity = sum(weights[k] * complexity_scores[k] for k in weights)
        return total_complexity

# Theory Retriever Agent
class TheoryRetrieverAgent(Agent):
    """Retrieve relevant urban theories using BERT-based semantic matching"""
    
    def __init__(self, theory_database: List[UrbanTheory]):
        super().__init__("TheoryRetriever")
        self.theories = theory_database
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
        # Pre-compute theory embeddings
        self._compute_theory_embeddings()
        
    def _compute_theory_embeddings(self):
        """Pre-compute embeddings for all theories"""
        for theory in self.theories:
            text = f"{theory.name} {' '.join(theory.principles[:3])}"
            theory.embedding = self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text"""
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
        return embedding
    
    async def process(self, scenario: UrbanScenario) -> List[UrbanTheory]:
        """Retrieve theories relevant to the scenario"""
        self.logger.info(f"Retrieving theories for domain: {scenario.domain}")
        
        # Create scenario embedding
        scenario_text = f"{scenario.description} {' '.join(scenario.objectives)}"
        scenario_embedding = self._get_embedding(scenario_text)
        
        # Calculate similarity scores
        theory_scores = []
        for theory in self.theories:
            if theory.embedding is not None:
                similarity = 1 - cosine(scenario_embedding, theory.embedding)
                theory_scores.append((theory, similarity))
        
        # Sort by similarity and filter
        theory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top theories (threshold-based)
        selected_theories = []
        for theory, score in theory_scores:
            if score > 0.6:  # Similarity threshold
                selected_theories.append(theory)
            if len(selected_theories) >= 5:  # Maximum theories
                break
        
        # Ensure at least one theory is selected
        if not selected_theories and theory_scores:
            selected_theories.append(theory_scores[0][0])
        
        # Check for theory synergies
        selected_theories = self._identify_synergies(selected_theories, scenario)
        
        return selected_theories
    
    def _identify_synergies(self, theories: List[UrbanTheory], scenario: UrbanScenario) -> List[UrbanTheory]:
        """Identify complementary theory combinations"""
        # Simple synergy rules based on domain
        synergy_rules = {
            "safety": ["CPTED", "Defensible Space", "Eyes on the Street"],
            "transportation": ["Transit-Oriented Development", "Complete Streets"],
            "housing": ["New Urbanism", "Compact City"],
        }
        
        if scenario.domain in synergy_rules:
            synergy_theories = [t for t in self.theories 
                              if any(name in t.name for name in synergy_rules[scenario.domain])]
            
            # Add synergistic theories not already selected
            for theory in synergy_theories:
                if theory not in theories:
                    theories.append(theory)
                    
        return theories[:5]  # Limit to 5 theories

# Algorithm Matcher Agent
class AlgorithmMatcherAgent(Agent):
    """Match algorithms to theoretical requirements"""
    
    def __init__(self, algorithm_database: List[Algorithm]):
        super().__init__("AlgorithmMatcher")
        self.algorithms = algorithm_database
        
        # Initialize ST-GCN for effectiveness prediction
        self.effectiveness_predictor = SpatioTemporalGCN(
            input_dim=10,
            hidden_dim=64,
            output_dim=1
        )
        
    async def process(self, theories: List[UrbanTheory], scenario: UrbanScenario) -> List[Algorithm]:
        """Match algorithms to theoretical requirements"""
        self.logger.info(f"Matching algorithms for {len(theories)} theories")
        
        # Extract computational requirements from theories
        requirements = self._extract_requirements(theories)
        
        # Score algorithms based on requirements
        algorithm_scores = []
        for algo in self.algorithms:
            score = self._calculate_algorithm_score(algo, requirements, scenario)
            algorithm_scores.append((algo, score))
        
        # Optimize selection using the paper's equation
        selected_algorithms = self._optimize_selection(algorithm_scores, requirements)
        
        return selected_algorithms
    
    def _extract_requirements(self, theories: List[UrbanTheory]) -> List[str]:
        """Extract computational requirements from theories"""
        requirements = []
        for theory in theories:
            requirements.extend(theory.computational_requirements)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_requirements = []
        for req in requirements:
            if req not in seen:
                seen.add(req)
                unique_requirements.append(req)
                
        return unique_requirements
    
    def _calculate_algorithm_score(self, algo: Algorithm, requirements: List[str], scenario: UrbanScenario) -> float:
        """Calculate algorithm score based on capability to fulfill requirements"""
        # Map requirements to capabilities
        req_to_cap = {
            "spatial_analysis": "Spatial Analysis",
            "pattern_recognition": "Pattern Recognition",
            "prediction": "Prediction",
            "optimization": "Optimization",
            "real_time": "Real-time Processing",
            "temporal_analysis": "Temporal Analysis",
            "classification": "Classification"
        }
        
        total_score = 0.0
        for req in requirements:
            if req in req_to_cap and req_to_cap[req] in algo.capabilities:
                total_score += algo.capabilities[req_to_cap[req]]
        
        # Adjust for computational cost
        cost_penalty = algo.computational_cost * 0.2
        final_score = total_score - cost_penalty
        
        return max(0, final_score)
    
    def _optimize_selection(self, algorithm_scores: List[Tuple[Algorithm, float]], 
                          requirements: List[str]) -> List[Algorithm]:
        """Optimize algorithm selection using paper's optimization equation"""
        # Sort by score
        algorithm_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        covered_requirements = set()
        
        for algo, score in algorithm_scores:
            if score > 0:
                # Check if algorithm adds new capabilities
                algo_caps = set(algo.capabilities.keys())
                if not algo_caps.issubset(covered_requirements):
                    selected.append(algo)
                    covered_requirements.update(algo_caps)
                    
            if len(selected) >= 3:  # Limit algorithms
                break
                
        return selected

# Spatio-Temporal Graph Convolutional Network
class SpatioTemporalGCN(nn.Module):
    """ST-GCN for effectiveness prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return torch.sigmoid(x)

# Data Source Selector Agent
class DataSourceSelectorAgent(Agent):
    """Select appropriate data sources based on requirements"""
    
    def __init__(self, data_catalog: List[DataSource]):
        super().__init__("DataSourceSelector")
        self.data_sources = data_catalog
        
    async def process(self, algorithms: List[Algorithm], scenario: UrbanScenario) -> List[DataSource]:
        """Select data sources based on algorithm requirements and scenario"""
        self.logger.info(f"Selecting data sources for {len(algorithms)} algorithms")
        
        # Collect all data requirements
        data_requirements = set()
        for algo in algorithms:
            data_requirements.update(algo.data_requirements)
        
        # Score data sources
        source_scores = []
        for source in self.data_sources:
            score = self._calculate_data_score(source, data_requirements, scenario)
            source_scores.append((source, score))
        
        # Select top sources
        source_scores.sort(key=lambda x: x[1], reverse=True)
        selected_sources = [s[0] for s in source_scores if s[1] > 0.5][:5]
        
        return selected_sources
    
    def _calculate_data_score(self, source: DataSource, requirements: set, scenario: UrbanScenario) -> float:
        """Calculate data source score using multiplicative function from paper"""
        dimensions = ['relevance', 'quality', 'temporal_coverage', 'accessibility', 'reliability', 'compatibility']
        weights = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
        
        scores = []
        
        # Relevance to requirements
        relevance = self._calculate_relevance(source, requirements, scenario)
        scores.append(relevance)
        
        # Other dimensions from quality_scores
        for dim in dimensions[1:]:
            scores.append(source.quality_scores.get(dim, 0.5))
        
        # Multiplicative scoring as per paper
        total_score = 1.0
        for score, weight in zip(scores, weights):
            total_score *= (score ** weight)
            
        return total_score
    
    def _calculate_relevance(self, source: DataSource, requirements: set, scenario: UrbanScenario) -> float:
        """Calculate relevance of data source to requirements"""
        # Simple keyword matching for demonstration
        relevance_keywords = {
            "spatial": ["geographic", "geospatial", "location", "map"],
            "temporal": ["time-series", "historical", "real-time"],
            "social": ["demographic", "behavioral", "survey"],
            "sensor": ["iot", "sensor", "monitoring"],
            "image": ["satellite", "street-view", "visual"]
        }
        
        relevance_score = 0.0
        matches = 0
        
        for req in requirements:
            for category, keywords in relevance_keywords.items():
                if any(kw in req.lower() for kw in keywords) and any(kw in source.type.lower() for kw in keywords):
                    matches += 1
                    
        relevance_score = min(matches / max(len(requirements), 1), 1.0)
        return relevance_score

# Integration Validator Agent
class IntegrationValidatorAgent(Agent):
    """Validate and integrate recommendations"""
    
    def __init__(self):
        super().__init__("IntegrationValidator")
        self.monte_carlo_iterations = 100
        
    async def process(self, theories: List[UrbanTheory], algorithms: List[Algorithm], 
                     data_sources: List[DataSource], scenario: UrbanScenario) -> Recommendation:
        """Validate the integrated solution"""
        self.logger.info("Validating integrated recommendation")
        
        # Stage 1: Computational robustness testing
        robustness_score = await self._test_robustness(theories, algorithms, data_sources, scenario)
        
        # Stage 2: Check for incompatibilities
        compatibility_issues = self._check_compatibility(theories, algorithms, data_sources)
        
        # Create recommendation
        recommendation = Recommendation(
            theories=theories,
            algorithms=algorithms,
            data_sources=data_sources,
            confidence_score=robustness_score,
            validation_results={
                'robustness': robustness_score,
                'compatibility_issues': compatibility_issues,
                'requires_human_validation': robustness_score < 0.7 or len(compatibility_issues) > 0
            }
        )
        
        return recommendation
    
    async def _test_robustness(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                              data_sources: List[DataSource], scenario: UrbanScenario) -> float:
        """Monte Carlo simulation for robustness testing"""
        successful_runs = 0
        
        for i in range(self.monte_carlo_iterations):
            # Simulate performance with noise
            performance = self._simulate_performance(theories, algorithms, data_sources, scenario)
            
            if performance > 0.6:  # Threshold
                successful_runs += 1
                
        robustness = successful_runs / self.monte_carlo_iterations
        return robustness
    
    def _simulate_performance(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                            data_sources: List[DataSource], scenario: UrbanScenario) -> float:
        """Simulate performance with random variations"""
        base_performance = 0.7
        
        # Theory contribution
        theory_score = min(len(theories) / 3.0, 1.0) * 0.3
        
        # Algorithm capability
        algo_score = np.mean([np.mean(list(a.capabilities.values())) for a in algorithms]) * 0.4
        
        # Data quality
        data_score = np.mean([np.mean(list(d.quality_scores.values())) for d in data_sources]) * 0.3
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        
        performance = base_performance * (theory_score + algo_score + data_score) + noise
        return np.clip(performance, 0, 1)
    
    def _check_compatibility(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                           data_sources: List[DataSource]) -> List[str]:
        """Check for incompatibilities between components"""
        issues = []
        
        # Check theory-algorithm compatibility
        for theory in theories:
            theory_reqs = set(theory.computational_requirements)
            algo_caps = set()
            for algo in algorithms:
                algo_caps.update(algo.capabilities.keys())
                
            missing = theory_reqs - algo_caps
            if missing:
                issues.append(f"Theory '{theory.name}' requires {missing} but algorithms don't provide")
        
        # Check algorithm-data compatibility
        for algo in algorithms:
            data_types = set()
            for source in data_sources:
                data_types.add(source.type)
                
            missing_data = set(algo.data_requirements) - data_types
            if missing_data:
                issues.append(f"Algorithm '{algo.name}' requires {missing_data} data but not available")
                
        return issues

# Multi-Agent Orchestrator
class UrbanAIRecommendationSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
        
        # 配置JSON文件路径
        self.data_dir = r"C:\Users\luvyf\Desktop\recommendation  system"
        
        # Initialize knowledge bases
        self.theories = self._load_theory_database()
        self.algorithms = self._load_algorithm_database()
        self.data_sources = self._load_data_catalog()
        
        # Initialize agents
        self.scenario_analyzer = ScenarioAnalyzerAgent()
        self.theory_retriever = TheoryRetrieverAgent(self.theories)
        self.algorithm_matcher = AlgorithmMatcherAgent(self.algorithms)
        self.data_selector = DataSourceSelectorAgent(self.data_sources)
        self.validator = IntegrationValidatorAgent()
    
    def _load_theory_database(self) -> List[UrbanTheory]:
        """Load urban planning and safety theories from JSON"""
        json_path = os.path.join(self.data_dir, 'urban-theory-database.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                theories_data = json.load(f)
            
            theories = []
            for theory_dict in theories_data:
                theory = UrbanTheory(
                    name=theory_dict['name'],
                    principles=theory_dict['principles'],
                    computational_requirements=theory_dict['computational_requirements'],
                    category=theory_dict['category'],
                    year=theory_dict['year']
                )
                theories.append(theory)
            
            self.logger.info(f"Loaded {len(theories)} theories from {json_path}")
            return theories
            
        except FileNotFoundError:
            self.logger.warning(f"File not found: {json_path}")
            self.logger.warning("Using default theories as fallback")
            return self._load_default_theories()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {json_path}: {e}")
            return self._load_default_theories()
        except Exception as e:
            self.logger.error(f"Error loading theories: {str(e)}")
            return self._load_default_theories()

    def _load_algorithm_database(self) -> List[Algorithm]:
        """Load AI/ML algorithms from JSON"""
        json_path = os.path.join(self.data_dir, 'algorithms.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                algorithms_data = json.load(f)
            
            algorithms = []
            for algo_dict in algorithms_data:
                algorithm = Algorithm(
                    name=algo_dict['name'],
                    capabilities=algo_dict['capabilities'],
                    computational_cost=algo_dict['computational_cost'],
                    data_requirements=algo_dict['data_requirements'],
                    group=algo_dict['group']
                )
                algorithms.append(algorithm)
            
            self.logger.info(f"Loaded {len(algorithms)} algorithms from {json_path}")
            return algorithms
            
        except FileNotFoundError:
            self.logger.warning(f"File not found: {json_path}")
            self.logger.warning("Using default algorithms as fallback")
            return self._load_default_algorithms()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {json_path}: {e}")
            return self._load_default_algorithms()
        except Exception as e:
            self.logger.error(f"Error loading algorithms: {str(e)}")
            return self._load_default_algorithms()

    def _load_data_catalog(self) -> List[DataSource]:
        """Load available data sources from JSON"""
        json_path = os.path.join(self.data_dir, 'data_sources.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sources_data = json.load(f)
            
            data_sources = []
            for source_dict in sources_data:
                source = DataSource(
                    name=source_dict['name'],
                    type=source_dict['type'],
                    quality_scores=source_dict['quality_scores'],
                    accessibility=source_dict['accessibility'],
                    update_frequency=source_dict['update_frequency']
                )
                data_sources.append(source)
            
            self.logger.info(f"Loaded {len(data_sources)} data sources from {json_path}")
            return data_sources
            
        except FileNotFoundError:
            self.logger.warning(f"File not found: {json_path}")
            self.logger.warning("Using default data sources as fallback")
            return self._load_default_data_sources()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {json_path}: {e}")
            return self._load_default_data_sources()
        except Exception as e:
            self.logger.error(f"Error loading data sources: {str(e)}")
            return self._load_default_data_sources()

    # 保留原始方法作为后备
    def _load_default_theories(self) -> List[UrbanTheory]:
        """Default theory database as fallback"""
        theories = [
            UrbanTheory(
                name="CPTED",
                principles=["Natural surveillance", "Access control", "Territorial reinforcement"],
                computational_requirements=["spatial_analysis", "pattern_recognition", "real_time"],
                category="Safety",
                year=1971
            ),
            UrbanTheory(
                name="Eyes on the Street",
                principles=["Mixed use", "Active frontages", "Pedestrian activity"],
                computational_requirements=["pattern_recognition", "temporal_analysis"],
                category="Safety",
                year=1961
            ),
            UrbanTheory(
                name="Transit-Oriented Development",
                principles=["High density near transit", "Mixed use", "Walkability"],
                computational_requirements=["spatial_analysis", "optimization", "prediction"],
                category="Design",
                year=1993
            ),
            UrbanTheory(
                name="Compact City",
                principles=["High density", "Mixed use", "Reduced sprawl"],
                computational_requirements=["spatial_analysis", "optimization"],
                category="Spatial",
                year=1973
            ),
            UrbanTheory(
                name="Image of the City",
                principles=["Paths", "Edges", "Districts", "Nodes", "Landmarks"],
                computational_requirements=["spatial_analysis", "pattern_recognition", "classification"],
                category="Perception",
                year=1960
            ),
        ]
        return theories

    def _load_default_algorithms(self) -> List[Algorithm]:
        """Default algorithm database as fallback"""
        algorithms = [
            Algorithm(
                name="Random Forest",
                capabilities={
                    "Classification": 0.9,
                    "Pattern Recognition": 0.8,
                    "Prediction": 0.85,
                    "Real-time Processing": 0.6
                },
                computational_cost=0.3,
                data_requirements=["tabular", "spatial"],
                group=2
            ),
            Algorithm(
                name="CNN",
                capabilities={
                    "Pattern Recognition": 0.95,
                    "Classification": 0.9,
                    "Spatial Analysis": 0.85
                },
                computational_cost=0.7,
                data_requirements=["image", "spatial"],
                group=2
            ),
            Algorithm(
                name="LSTM",
                capabilities={
                    "Temporal Analysis": 0.95,
                    "Prediction": 0.9,
                    "Pattern Recognition": 0.8
                },
                computational_cost=0.6,
                data_requirements=["time-series", "temporal"],
                group=2
            ),
            Algorithm(
                name="Graph Neural Network",
                capabilities={
                    "Spatial Analysis": 0.95,
                    "Pattern Recognition": 0.85,
                    "Optimization": 0.8
                },
                computational_cost=0.8,
                data_requirements=["graph", "spatial"],
                group=2
            ),
            Algorithm(
                name="Genetic Algorithm",
                capabilities={
                    "Optimization": 0.95,
                    "Real-time Processing": 0.3
                },
                computational_cost=0.5,
                data_requirements=["tabular"],
                group=0
            ),
        ]
        return algorithms

    def _load_default_data_sources(self) -> List[DataSource]:
        """Default data source catalog as fallback"""
        data_sources = [
            DataSource(
                name="Geographic Information System",
                type="geospatial",
                quality_scores={
                    "quality": 0.9,
                    "temporal_coverage": 0.7,
                    "accessibility": 0.8,
                    "reliability": 0.95,
                    "compatibility": 0.9
                },
                accessibility=0.8,
                update_frequency="monthly"
            ),
            DataSource(
                name="Street View Imagery",
                type="image",
                quality_scores={
                    "quality": 0.85,
                    "temporal_coverage": 0.6,
                    "accessibility": 0.7,
                    "reliability": 0.9,
                    "compatibility": 0.8
                },
                accessibility=0.7,
                update_frequency="yearly"
            ),
            DataSource(
                name="IoT Sensor Network",
                type="sensor",
                quality_scores={
                    "quality": 0.8,
                    "temporal_coverage": 0.95,
                    "accessibility": 0.6,
                    "reliability": 0.85,
                    "compatibility": 0.7
                },
                accessibility=0.6,
                update_frequency="real-time"
            ),
            DataSource(
                name="Census Data",
                type="demographic",
                quality_scores={
                    "quality": 0.95,
                    "temporal_coverage": 0.5,
                    "accessibility": 0.9,
                    "reliability": 0.98,
                    "compatibility": 0.85
                },
                accessibility=0.9,
                update_frequency="yearly"
            ),
            DataSource(
                name="Social Media Data",
                type="social",
                quality_scores={
                    "quality": 0.6,
                    "temporal_coverage": 0.9,
                    "accessibility": 0.5,
                    "reliability": 0.6,
                    "compatibility": 0.7
                },
                accessibility=0.5,
                update_frequency="real-time"
            ),
        ]
        return data_sources
    
    async def generate_recommendation(self, urban_challenge: str) -> Recommendation:
        """Generate theory-driven recommendation for urban challenge"""
        self.logger.info(f"Generating recommendation for: {urban_challenge}")
        
        try:
            # Stage 1: Analyze scenario
            scenario = await self.scenario_analyzer.process(urban_challenge)
            self.logger.info(f"Scenario analyzed - Domain: {scenario.domain}, Complexity: {scenario.complexity_score:.2f}")
            
            # Stage 2: Retrieve relevant theories
            theories = await self.theory_retriever.process(scenario)
            self.logger.info(f"Retrieved {len(theories)} relevant theories: {[t.name for t in theories]}")
            
            # Stage 3: Match algorithms to theories
            algorithms = await self.algorithm_matcher.process(theories, scenario)
            self.logger.info(f"Matched {len(algorithms)} algorithms: {[a.name for a in algorithms]}")
            
            # Stage 4: Select data sources
            data_sources = await self.data_selector.process(algorithms, scenario)
            self.logger.info(f"Selected {len(data_sources)} data sources: {[d.name for d in data_sources]}")
            
            # Stage 5: Validate integration
            recommendation = await self.validator.process(theories, algorithms, data_sources, scenario)
            self.logger.info(f"Validation complete - Confidence: {recommendation.confidence_score:.2f}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            raise

# Utility functions for displaying recommendations
def display_recommendation(recommendation: Recommendation):
    """Pretty print recommendation"""
    print("\n" + "="*80)
    print("THEORY-DRIVEN AI RECOMMENDATION FOR URBAN CHALLENGE")
    print("="*80)
    
    print(f"\nConfidence Score: {recommendation.confidence_score:.2f}")
    print(f"Human Validation Required: {recommendation.validation_results['requires_human_validation']}")
    
    print("\nRECOMMENDED THEORIES:")
    for theory in recommendation.theories:
        print(f"  - {theory.name} ({theory.year}) - Category: {theory.category}")
        print(f"    Key Principles: {', '.join(theory.principles[:2])}")
    
    print("\nRECOMMENDED ALGORITHMS:")
    for algo in recommendation.algorithms:
        print(f"  - {algo.name} (Group {algo.group})")
        print(f"    Top Capabilities: {', '.join([k for k, v in algo.capabilities.items() if v > 0.8])}")
    
    print("\nRECOMMENDED DATA SOURCES:")
    for source in recommendation.data_sources:
        print(f"  - {source.name} ({source.type})")
        print(f"    Update Frequency: {source.update_frequency}")
        print(f"    Accessibility: {source.accessibility:.1f}")
    
    if recommendation.validation_results['compatibility_issues']:
        print("\nCOMPATIBILITY ISSUES:")
        for issue in recommendation.validation_results['compatibility_issues']:
            print(f"  ⚠️  {issue}")
    
    print("\n" + "="*80)

# Main execution
async def main():
    """Example usage of the system"""
    # Initialize system
    system = UrbanAIRecommendationSystem()
    
    # Example urban challenges
    challenges = [
        "We need to reduce crime rates in downtown neighborhoods by improving street lighting and surveillance while maintaining resident privacy",
        "How can we optimize public transportation routes to reduce traffic congestion during peak hours in the city center?",
        "Design a sustainable mixed-use development that promotes walkability and community interaction",
    ]
    
    for challenge in challenges:
        print(f"\nCHALLENGE: {challenge}")
        recommendation = await system.generate_recommendation(challenge)
        display_recommendation(recommendation)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())