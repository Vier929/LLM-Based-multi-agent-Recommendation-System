"""
Theory-Driven Multi-Agent Recommendation System for Urban AI Applications
A lightweight demo implementation optimized for cloud deployment
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import os
import re
from collections import defaultdict

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

# Lightweight text embedding class (replaces BERT)
class LightweightTextEmbedding:
    """Lightweight text embedding using TF-IDF and word vectors"""
    
    def __init__(self):
        # Pre-computed urban domain vocabulary with embeddings
        self.vocab = {
            'traffic': np.array([0.1, 0.8, 0.2, 0.4, 0.7]),
            'crime': np.array([0.9, 0.1, 0.8, 0.3, 0.2]),
            'safety': np.array([0.8, 0.2, 0.9, 0.4, 0.3]),
            'transportation': np.array([0.2, 0.9, 0.3, 0.8, 0.6]),
            'housing': np.array([0.3, 0.4, 0.2, 0.9, 0.7]),
            'development': np.array([0.4, 0.6, 0.3, 0.8, 0.8]),
            'urban': np.array([0.5, 0.7, 0.6, 0.7, 0.8]),
            'planning': np.array([0.6, 0.5, 0.4, 0.9, 0.7]),
            'design': np.array([0.7, 0.3, 0.5, 0.6, 0.9]),
            'spatial': np.array([0.3, 0.8, 0.4, 0.5, 0.6]),
            'surveillance': np.array([0.8, 0.2, 0.7, 0.3, 0.4]),
            'walkability': np.array([0.2, 0.7, 0.3, 0.8, 0.9]),
            'sustainability': np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
            'community': np.array([0.5, 0.6, 0.7, 0.8, 0.7]),
            'mixed': np.array([0.6, 0.4, 0.5, 0.7, 0.6]),
            'density': np.array([0.3, 0.5, 0.4, 0.6, 0.7]),
            'accessibility': np.array([0.4, 0.7, 0.5, 0.8, 0.6]),
            'efficiency': np.array([0.5, 0.8, 0.6, 0.7, 0.5]),
            'optimization': np.array([0.7, 0.9, 0.5, 0.6, 0.4]),
            'prediction': np.array([0.8, 0.6, 0.7, 0.4, 0.5])
        }
        self.embedding_dim = 5
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using lightweight approach"""
        words = re.findall(r'\w+', text.lower())
        embeddings = []
        
        for word in words:
            if word in self.vocab:
                embeddings.append(self.vocab[word])
            else:
                # Generate pseudo-embedding for unknown words
                hash_val = hash(word) % 1000
                embedding = np.random.RandomState(hash_val).uniform(-0.5, 0.5, self.embedding_dim)
                embeddings.append(embedding)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

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
        self.text_embedder = LightweightTextEmbedding()
        
        # Domain keywords for classification
        self.domain_keywords = {
            "transportation": ["traffic", "transport", "mobility", "transit", "vehicle", "parking", "road"],
            "safety": ["crime", "security", "surveillance", "safety", "emergency", "police"],
            "housing": ["housing", "residential", "dwelling", "affordable", "gentrification", "development"],
            "environment": ["pollution", "green", "sustainability", "climate", "emission", "energy"],
            "economic": ["business", "economy", "employment", "commerce", "retail", "economic"]
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
        """Extract structured information using pattern matching"""
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
        objective_patterns = ["to reduce", "to improve", "to enhance", "to optimize", "to minimize", "to increase", "to develop"]
        objectives = []
        
        for pattern in objective_patterns:
            if pattern in text.lower():
                start = text.lower().find(pattern)
                end = min(start + 60, len(text))
                objective = text[start:end].strip()
                if objective not in objectives:
                    objectives.append(objective)
                
        return objectives[:3]
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from text"""
        constraint_patterns = ["limited", "constraint", "restriction", "budget", "within", "while maintaining", "without"]
        constraints = []
        
        for pattern in constraint_patterns:
            if pattern in text.lower():
                start = text.lower().find(pattern)
                end = min(start + 50, len(text))
                constraint = text[start:end].strip()
                if constraint not in constraints:
                    constraints.append(constraint)
                
        return constraints[:3]
    
    def _extract_stakeholders(self, text: str) -> List[str]:
        """Extract stakeholders mentioned in text"""
        stakeholder_keywords = ["residents", "government", "businesses", "community", "citizens",
                               "authorities", "developers", "organizations", "agencies", "planners"]
        
        stakeholders = [kw for kw in stakeholder_keywords if kw in text.lower()]
        return list(set(stakeholders))[:4]
    
    def _extract_temporal(self, text: str) -> str:
        """Extract temporal scope"""
        if any(word in text.lower() for word in ["immediate", "urgent", "now", "asap"]):
            return "immediate"
        elif any(word in text.lower() for word in ["short-term", "months", "quarterly"]):
            return "short-term"
        elif any(word in text.lower() for word in ["long-term", "years", "decade"]):
            return "long-term"
        else:
            return "medium-term"
    
    def _extract_spatial(self, text: str) -> str:
        """Extract spatial boundaries"""
        if any(word in text.lower() for word in ["neighborhood", "district", "block"]):
            return "neighborhood"
        elif any(word in text.lower() for word in ["city", "urban", "downtown", "center"]):
            return "city"
        elif any(word in text.lower() for word in ["region", "metropolitan", "county"]):
            return "regional"
        else:
            return "city"
    
    def _calculate_complexity(self, scenario: UrbanScenario) -> float:
        """Calculate scenario complexity score"""
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
            'temporal': 0.9 if scenario.temporal_scope == "immediate" else 0.6 if scenario.temporal_scope == "short-term" else 0.8,
            'spatial': 0.4 if scenario.spatial_boundaries == "neighborhood" else 0.7 if scenario.spatial_boundaries == "city" else 1.0,
            'domain_complexity': 0.8 if scenario.domain in ["safety", "transportation"] else 0.6
        }
        
        total_complexity = sum(weights[k] * complexity_scores[k] for k in weights)
        return min(total_complexity, 1.0)

# Theory Retriever Agent
class TheoryRetrieverAgent(Agent):
    """Retrieve relevant urban theories using semantic matching"""
    
    def __init__(self, theory_database: List[UrbanTheory]):
        super().__init__("TheoryRetriever")
        self.theories = theory_database
        self.text_embedder = LightweightTextEmbedding()
        
        # Pre-compute theory embeddings
        self._compute_theory_embeddings()
        
    def _compute_theory_embeddings(self):
        """Pre-compute embeddings for all theories"""
        for theory in self.theories:
            text = f"{theory.name} {' '.join(theory.principles[:3])}"
            theory.embedding = self.text_embedder.get_embedding(text)
    
    async def process(self, scenario: UrbanScenario) -> List[UrbanTheory]:
        """Retrieve theories relevant to the scenario"""
        self.logger.info(f"Retrieving theories for domain: {scenario.domain}")
        
        # Create scenario embedding
        scenario_text = f"{scenario.description} {' '.join(scenario.objectives)}"
        scenario_embedding = self.text_embedder.get_embedding(scenario_text)
        
        # Calculate similarity scores
        theory_scores = []
        for theory in self.theories:
            if theory.embedding is not None:
                similarity = 1 - cosine(scenario_embedding, theory.embedding)
                # Add domain-specific boost
                if self._is_domain_relevant(theory, scenario.domain):
                    similarity += 0.2
                theory_scores.append((theory, similarity))
        
        # Sort by similarity and filter
        theory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top theories
        selected_theories = []
        for theory, score in theory_scores:
            if score > 0.4:  # Lower threshold for demo
                selected_theories.append(theory)
            if len(selected_theories) >= 5:
                break
        
        # Ensure at least 2 theories are selected
        if len(selected_theories) < 2 and theory_scores:
            for theory, _ in theory_scores[:2]:
                if theory not in selected_theories:
                    selected_theories.append(theory)
        
        # Add complementary theories
        selected_theories = self._identify_synergies(selected_theories, scenario)
        
        return selected_theories[:5]
    
    def _is_domain_relevant(self, theory: UrbanTheory, domain: str) -> bool:
        """Check if theory is relevant to domain"""
        domain_theory_map = {
            "safety": ["Safety", "Design"],
            "transportation": ["Design", "Spatial"],
            "housing": ["Design", "Spatial"],
            "environment": ["Design", "Spatial"],
            "economic": ["Design", "Spatial"]
        }
        
        return theory.category in domain_theory_map.get(domain, [])
    
    def _identify_synergies(self, theories: List[UrbanTheory], scenario: UrbanScenario) -> List[UrbanTheory]:
        """Identify complementary theory combinations"""
        synergy_rules = {
            "safety": ["CPTED", "Defensible Space", "Eyes on the Street"],
            "transportation": ["Transit-Oriented Development", "Complete Streets"],
            "housing": ["New Urbanism", "Compact City"],
        }
        
        if scenario.domain in synergy_rules:
            for theory in self.theories:
                if (any(name in theory.name for name in synergy_rules[scenario.domain]) 
                    and theory not in theories and len(theories) < 4):
                    theories.append(theory)
                    
        return theories

# Algorithm Matcher Agent  
class AlgorithmMatcherAgent(Agent):
    """Match algorithms to theoretical requirements"""
    
    def __init__(self, algorithm_database: List[Algorithm]):
        super().__init__("AlgorithmMatcher")
        self.algorithms = algorithm_database
        
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
        
        # Optimize selection
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
        matched_requirements = 0
        
        for req in requirements:
            if req in req_to_cap and req_to_cap[req] in algo.capabilities:
                total_score += algo.capabilities[req_to_cap[req]]
                matched_requirements += 1
        
        # Normalize by number of requirements
        if matched_requirements > 0:
            total_score = total_score / matched_requirements
        
        # Domain-specific bonuses
        if scenario.domain == "safety" and "Pattern Recognition" in algo.capabilities:
            total_score += 0.1
        if scenario.domain == "transportation" and "Optimization" in algo.capabilities:
            total_score += 0.1
            
        # Adjust for computational cost (prefer efficient algorithms)
        cost_penalty = algo.computational_cost * 0.1
        final_score = max(0, total_score - cost_penalty)
        
        return final_score
    
    def _optimize_selection(self, algorithm_scores: List[Tuple[Algorithm, float]], 
                          requirements: List[str]) -> List[Algorithm]:
        """Optimize algorithm selection"""
        algorithm_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        covered_capabilities = set()
        
        for algo, score in algorithm_scores:
            if score > 0.3:  # Lower threshold for demo
                algo_caps = set(algo.capabilities.keys())
                
                # Add if provides new capabilities or score is very high
                if not algo_caps.issubset(covered_capabilities) or score > 0.8:
                    selected.append(algo)
                    covered_capabilities.update(algo_caps)
                    
            if len(selected) >= 4:  # Slightly more algorithms for better demo
                break
                
        # Ensure at least 2 algorithms
        if len(selected) < 2:
            for algo, _ in algorithm_scores[:2]:
                if algo not in selected:
                    selected.append(algo)
                    
        return selected

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
        selected_sources = [s[0] for s in source_scores if s[1] > 0.3][:6]  # More sources for demo
        
        # Ensure diverse data types
        selected_sources = self._ensure_diversity(selected_sources, data_requirements)
        
        return selected_sources
    
    def _calculate_data_score(self, source: DataSource, requirements: set, scenario: UrbanScenario) -> float:
        """Calculate data source score"""
        dimensions = ['relevance', 'quality', 'temporal_coverage', 'accessibility', 'reliability', 'compatibility']
        weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        
        scores = []
        
        # Relevance to requirements
        relevance = self._calculate_relevance(source, requirements, scenario)
        scores.append(relevance)
        
        # Other dimensions from quality_scores
        for dim in dimensions[1:]:
            scores.append(source.quality_scores.get(dim, 0.5))
        
        # Weighted average instead of multiplicative for more forgiving scoring
        total_score = sum(score * weight for score, weight in zip(scores, weights))
        
        # Domain-specific boosts
        if scenario.domain == "safety" and "sensor" in source.type.lower():
            total_score += 0.1
        if scenario.domain == "transportation" and "spatial" in source.type.lower():
            total_score += 0.1
            
        return min(total_score, 1.0)
    
    def _calculate_relevance(self, source: DataSource, requirements: set, scenario: UrbanScenario) -> float:
        """Calculate relevance of data source to requirements"""
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
                if (any(kw in req.lower() for kw in keywords) and 
                    any(kw in source.type.lower() for kw in keywords)):
                    matches += 1
                    
        if requirements:
            relevance_score = min(matches / len(requirements), 1.0)
        else:
            relevance_score = 0.5
            
        # Boost for domain relevance
        domain_boosts = {
            "safety": ["sensor", "social", "image"],
            "transportation": ["spatial", "sensor", "temporal"],
            "housing": ["demographic", "spatial"],
            "environment": ["sensor", "temporal", "spatial"]
        }
        
        if scenario.domain in domain_boosts:
            for boost_type in domain_boosts[scenario.domain]:
                if boost_type in source.type.lower():
                    relevance_score += 0.2
                    
        return min(relevance_score, 1.0)
    
    def _ensure_diversity(self, sources: List[DataSource], requirements: set) -> List[DataSource]:
        """Ensure diversity in selected data sources"""
        if len(sources) < 3:
            return sources
            
        # Group by type
        type_groups = defaultdict(list)
        for source in sources:
            type_groups[source.type].append(source)
        
        # Select best from each type
        diverse_sources = []
        for source_type, type_sources in type_groups.items():
            # Sort by quality and take best
            type_sources.sort(key=lambda s: np.mean(list(s.quality_scores.values())), reverse=True)
            diverse_sources.append(type_sources[0])
            
        return diverse_sources[:5]

# Integration Validator Agent
class IntegrationValidatorAgent(Agent):
    """Validate and integrate recommendations"""
    
    def __init__(self):
        super().__init__("IntegrationValidator")
        self.monte_carlo_iterations = 50  # Reduced for faster demo
        
    async def process(self, theories: List[UrbanTheory], algorithms: List[Algorithm], 
                     data_sources: List[DataSource], scenario: UrbanScenario) -> Recommendation:
        """Validate the integrated solution"""
        self.logger.info("Validating integrated recommendation")
        
        # Stage 1: Computational robustness testing
        robustness_score = await self._test_robustness(theories, algorithms, data_sources, scenario)
        
        # Stage 2: Check for incompatibilities
        compatibility_issues = self._check_compatibility(theories, algorithms, data_sources)
        
        # Stage 3: Calculate confidence
        confidence_score = self._calculate_confidence(theories, algorithms, data_sources, scenario, robustness_score)
        
        # Create recommendation
        recommendation = Recommendation(
            theories=theories,
            algorithms=algorithms,
            data_sources=data_sources,
            confidence_score=confidence_score,
            validation_results={
                'robustness': robustness_score,
                'compatibility_issues': compatibility_issues,
                'requires_human_validation': confidence_score < 0.7 or len(compatibility_issues) > 0
            }
        )
        
        return recommendation
    
    async def _test_robustness(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                              data_sources: List[DataSource], scenario: UrbanScenario) -> float:
        """Simplified robustness testing"""
        successful_runs = 0
        
        for i in range(self.monte_carlo_iterations):
            performance = self._simulate_performance(theories, algorithms, data_sources, scenario)
            
            if performance > 0.5:  # Lower threshold for demo
                successful_runs += 1
                
        robustness = successful_runs / self.monte_carlo_iterations
        return robustness
    
    def _simulate_performance(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                            data_sources: List[DataSource], scenario: UrbanScenario) -> float:
        """Simulate performance with random variations"""
        # Base performance calculation
        theory_score = min(len(theories) / 3.0, 1.0) * 0.3
        
        if algorithms:
            algo_score = np.mean([np.mean(list(a.capabilities.values())) for a in algorithms]) * 0.4
        else:
            algo_score = 0.0
            
        if data_sources:
            data_score = np.mean([np.mean(list(d.quality_scores.values())) for d in data_sources]) * 0.3
        else:
            data_score = 0.0
        
        # Add small random variation
        noise = np.random.normal(0, 0.05)
        
        performance = theory_score + algo_score + data_score + noise
        return np.clip(performance, 0, 1)
    
    def _calculate_confidence(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                            data_sources: List[DataSource], scenario: UrbanScenario, robustness: float) -> float:
        """Calculate overall confidence score"""
        # Component scores
        theory_confidence = min(len(theories) / 3.0, 1.0)
        algo_confidence = min(len(algorithms) / 3.0, 1.0)
        data_confidence = min(len(data_sources) / 4.0, 1.0)
        
        # Complexity adjustment
        complexity_penalty = scenario.complexity_score * 0.1
        
        # Base confidence
        base_confidence = (theory_confidence * 0.3 + algo_confidence * 0.35 + 
                          data_confidence * 0.25 + robustness * 0.1)
        
        final_confidence = max(0, base_confidence - complexity_penalty)
        
        return min(final_confidence, 1.0)
    
    def _check_compatibility(self, theories: List[UrbanTheory], algorithms: List[Algorithm],
                           data_sources: List[DataSource]) -> List[str]:
        """Check for incompatibilities between components"""
        issues = []
        
        # Check theory-algorithm compatibility
        all_theory_reqs = set()
        for theory in theories:
            all_theory_reqs.update(theory.computational_requirements)
            
        all_algo_caps = set()
        for algo in algorithms:
            all_algo_caps.update(algo.capabilities.keys())
            
        # Map requirements to capabilities
        req_cap_map = {
            "spatial_analysis": "Spatial Analysis",
            "pattern_recognition": "Pattern Recognition",
            "prediction": "Prediction",
            "optimization": "Optimization",
            "real_time": "Real-time Processing",
            "temporal_analysis": "Temporal Analysis",
            "classification": "Classification"
        }
        
        for req in all_theory_reqs:
            if req in req_cap_map:
                cap = req_cap_map[req]
                if cap not in all_algo_caps:
                    issues.append(f"Theory requirement '{req}' not fully covered by selected algorithms")
        
        # Check algorithm-data compatibility
        all_data_types = set(ds.type for ds in data_sources)
        for algo in algorithms:
            for data_req in algo.data_requirements:
                if not any(data_req.lower() in dt.lower() for dt in all_data_types):
                    issues.append(f"Algorithm '{algo.name}' requires '{data_req}' data type not available")
                    
        return issues

# Multi-Agent Orchestrator
class UrbanAIRecommendationSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
        
        # Initialize knowledge bases with defaults (no file dependencies)
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
        """Load urban planning and safety theories"""
        theories = [
            UrbanTheory(
                name="Crime Prevention Through Environmental Design (CPTED)",
                principles=["Natural surveillance", "Access control", "Territorial reinforcement", "Maintenance"],
                computational_requirements=["spatial_analysis", "pattern_recognition", "real_time"],
                category="Safety",
                year=1971
            ),
            UrbanTheory(
                name="Eyes on the Street",
                principles=["Mixed use development", "Active street frontages", "Pedestrian activity", "Community presence"],
                computational_requirements=["pattern_recognition", "temporal_analysis"],
                category="Safety",
                year=1961
            ),
            UrbanTheory(
                name="Defensible Space Theory",
                principles=["Territorial definition", "Natural surveillance", "Image and milieu", "Safe adjacencies"],
                computational_requirements=["spatial_analysis", "classification"],
                category="Safety",
                year=1972
            ),
            UrbanTheory(
                name="Transit-Oriented Development",
                principles=["High density near transit", "Mixed land use", "Walkability", "Reduced parking"],
                computational_requirements=["spatial_analysis", "optimization", "prediction"],
                category="Design",
                year=1993
            ),
            UrbanTheory(
                name="New Urbanism",
                principles=["Walkable neighborhoods", "Mixed-use development", "Transit access", "Narrow streets"],
                computational_requirements=["spatial_analysis", "optimization"],
                category="Design",
                year=1980
            ),
            UrbanTheory(
                name="Compact City Theory",
                principles=["High density", "Mixed land use", "Reduced urban sprawl", "Sustainable transport"],
                computational_requirements=["spatial_analysis", "optimization"],
                category="Spatial",
                year=1973
            ),
            UrbanTheory(
                name="Smart Growth",
                principles=["Infill development", "Transit access", "Walkable communities", "Open space preservation"],
                computational_requirements=["spatial_analysis", "optimization", "prediction"],
                category="Spatial",
                year=1990
            ),
            UrbanTheory(
                name="Image of the City",
                principles=["Paths", "Edges", "Districts", "Nodes", "Landmarks"],
                computational_requirements=["spatial_analysis", "pattern_recognition", "classification"],
                category="Perception",
                year=1960
            ),
            UrbanTheory(
                name="Complete Streets",
                principles=["Multi-modal design", "Safety for all users", "Context-sensitive solutions", "Accessibility"],
                computational_requirements=["spatial_analysis", "optimization"],
                category="Design",
                year=2003
            ),
            UrbanTheory(
                name="Sustainable Urban Design",
                principles=["Energy efficiency", "Green infrastructure", "Resource conservation", "Climate adaptation"],
                computational_requirements=["optimization", "prediction", "temporal_analysis"],
                category="Design",
                year=1987
            ),
        ]
        return theories

    def _load_algorithm_database(self) -> List[Algorithm]:
        """Load AI/ML algorithms database"""
        algorithms = [
            Algorithm(
                name="Random Forest",
                capabilities={
                    "Classification": 0.9,
                    "Pattern Recognition": 0.8,
                    "Prediction": 0.85,
                    "Real-time Processing": 0.7
                },
                computational_cost=0.3,
                data_requirements=["tabular", "spatial"],
                group=2
            ),
            Algorithm(
                name="Convolutional Neural Network (CNN)",
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
                name="Long Short-Term Memory (LSTM)",
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
                name="Graph Neural Network (GNN)",
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
                name="Support Vector Machine (SVM)",
                capabilities={
                    "Classification": 0.85,
                    "Pattern Recognition": 0.8,
                    "Prediction": 0.75
                },
                computational_cost=0.4,
                data_requirements=["tabular"],
                group=1
            ),
            Algorithm(
                name="Genetic Algorithm",
                capabilities={
                    "Optimization": 0.95,
                    "Real-time Processing": 0.4
                },
                computational_cost=0.5,
                data_requirements=["tabular"],
                group=0
            ),
            Algorithm(
                name="K-Means Clustering",
                capabilities={
                    "Classification": 0.7,
                    "Pattern Recognition": 0.75,
                    "Spatial Analysis": 0.6
                },
                computational_cost=0.2,
                data_requirements=["tabular", "spatial"],
                group=1
            ),
            Algorithm(
                name="Reinforcement Learning",
                capabilities={
                    "Optimization": 0.9,
                    "Real-time Processing": 0.8,
                    "Prediction": 0.85
                },
                computational_cost=0.9,
                data_requirements=["temporal", "tabular"],
                group=3
            ),
            Algorithm(
                name="XGBoost",
                capabilities={
                    "Classification": 0.9,
                    "Prediction": 0.88,
                    "Pattern Recognition": 0.8,
                    "Real-time Processing": 0.6
                },
                computational_cost=0.4,
                data_requirements=["tabular"],
                group=2
            ),
            Algorithm(
                name="DBSCAN",
                capabilities={
                    "Classification": 0.75,
                    "Pattern Recognition": 0.8,
                    "Spatial Analysis": 0.85
                },
                computational_cost=0.3,
                data_requirements=["spatial", "tabular"],
                group=1
            ),
        ]
        return algorithms

    def _load_data_catalog(self) -> List[DataSource]:
        """Load available data sources catalog"""
        data_sources = [
            DataSource(
                name="Geographic Information System (GIS)",
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
                name="Census Demographics",
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
                name="Social Media Sentiment",
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
            DataSource(
                name="Traffic Flow Data",
                type="temporal",
                quality_scores={
                    "quality": 0.85,
                    "temporal_coverage": 0.9,
                    "accessibility": 0.7,
                    "reliability": 0.9,
                    "compatibility": 0.85
                },
                accessibility=0.7,
                update_frequency="real-time"
            ),
            DataSource(
                name="Building Footprints",
                type="spatial",
                quality_scores={
                    "quality": 0.9,
                    "temporal_coverage": 0.6,
                    "accessibility": 0.8,
                    "reliability": 0.95,
                    "compatibility": 0.9
                },
                accessibility=0.8,
                update_frequency="yearly"
            ),
            DataSource(
                name="Crime Incident Reports",
                type="tabular",
                quality_scores={
                    "quality": 0.8,
                    "temporal_coverage": 0.85,
                    "accessibility": 0.7,
                    "reliability": 0.9,
                    "compatibility": 0.8
                },
                accessibility=0.7,
                update_frequency="daily"
            ),
            DataSource(
                name="Environmental Monitoring",
                type="sensor",
                quality_scores={
                    "quality": 0.85,
                    "temporal_coverage": 0.9,
                    "accessibility": 0.6,
                    "reliability": 0.88,
                    "compatibility": 0.75
                },
                accessibility=0.6,
                update_frequency="hourly"
            ),
            DataSource(
                name="Public Transit Data",
                type="temporal",
                quality_scores={
                    "quality": 0.9,
                    "temporal_coverage": 0.95,
                    "accessibility": 0.8,
                    "reliability": 0.92,
                    "compatibility": 0.85
                },
                accessibility=0.8,
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
        top_caps = [k for k, v in algo.capabilities.items() if v > 0.7]
        print(f"    Top Capabilities: {', '.join(top_caps)}")
    
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

# Main execution for testing
async def main():
    """Example usage of the system"""
    system = UrbanAIRecommendationSystem()
    
    challenge = "We need to reduce crime rates in downtown neighborhoods by improving street lighting and surveillance while maintaining resident privacy"
    
    print(f"\nCHALLENGE: {challenge}")
    recommendation = await system.generate_recommendation(challenge)
    display_recommendation(recommendation)

if __name__ == "__main__":
    asyncio.run(main())