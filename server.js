// Direct API Keys (inline) - no .env file needed
console.log('🔑 Using inline API keys...');
const FLORENCE_API_KEY = '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';
const NVIDIA_API_KEY = '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';

console.log('✅ API Keys loaded directly:');
console.log('   Florence-2:', FLORENCE_API_KEY ? '✅ Present (' + FLORENCE_API_KEY.length + ' chars)' : '❌ Missing');
console.log('   NVIDIA:', NVIDIA_API_KEY ? '✅ Present (' + NVIDIA_API_KEY.length + ' chars)' : '❌ Missing');

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const axios = require('axios');
const sharp = require('sharp');
const FormData = require('form-data');

const app = express();
const PORT = 8000;

// API URLs
const FLORENCE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large";
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";

// Enhanced CORS configuration
app.use(cors({
  origin: true, // Allow all origins for development
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Handle preflight OPTIONS requests
app.options('*', cors());

// Middleware
app.use(express.json({ limit: '10mb' }));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url} - ${new Date().toISOString()}`);
  next();
});

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  }
});

// Simplified feature extraction using Florence-2 for basic image understanding
async function extractFeaturesWithFlorence(imageBuffer) {
  try {
    console.log('📤 Sending request to Florence-2...');
    console.log('📏 Image buffer size:', imageBuffer.length, 'bytes');
    
    // Convert image to base64 for Hugging Face API
    const base64Image = imageBuffer.toString('base64');
    console.log('🔄 Converted to base64, length:', base64Image.length);
    
    // Use the correct Hugging Face Inference API format
    const response = await axios.post(FLORENCE_API_URL, {
      inputs: base64Image,
      parameters: {
        task: "image-to-text"
      }
    }, {
      headers: {
        'Authorization': `Bearer ${FLORENCE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    console.log('📥 Florence-2 response received:', response.status);
    
    if (response.status === 200 && response.data) {
      console.log('🔍 Florence-2 data:', JSON.stringify(response.data, null, 2));
      
      // Extract meaningful text from Florence-2 response
      let description = '';
      if (Array.isArray(response.data) && response.data.length > 0) {
        description = response.data[0].generated_text || '';
      } else if (response.data.generated_text) {
        description = response.data.generated_text;
      } else if (typeof response.data === 'string') {
        description = response.data;
      } else if (response.data[0] && response.data[0].generated_text) {
        description = response.data[0].generated_text;
      }
      
      console.log('📝 Extracted description:', description);
      
      // Create features based on the description
      return createFeaturesFromDescription(description);
      
    } else {
      throw new Error(`Florence-2 API error: ${response.status}`);
    }
  } catch (error) {
    console.error('❌ Florence-2 extraction error:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response headers:', error.response.headers);
      console.error('Response data:', JSON.stringify(error.response.data, null, 2));
    }
    return getFallbackFeatures();
  }
}

// Create structured features from Florence-2 description using GeoGuessr framework
function createFeaturesFromDescription(description) {
  console.log('🔧 Creating GeoGuessr features from description:', description);
  
  const lowerDesc = description.toLowerCase();
  
  // Build comprehensive GeoGuessr feature structure
  const features = {
    sun_dir: "unknown",
    cam_gen: "unknown", 
    drive_side: "unknown",
    road: {
      lines: "unknown",
      surface: detectSurface(lowerDesc),
      shoulder: "unknown",
      median: "unknown", 
      curvature: detectCurvature(lowerDesc),
      elevation: detectElevation(lowerDesc)
    },
    bollards: "unknown",
    poles: detectPoles(lowerDesc),
    guardrails: "unknown",
    signs: {
      lang: "unknown",
      shapes: "unknown",
      units: "unknown",
      mounts: "unknown"
    },
    license_plate: {
      front: "unknown",
      rear: "unknown", 
      blur_status: "unknown",
      country_code: "unknown"
    },
    text_features: {
      language: detectLanguage(lowerDesc),
      toponyms: "unknown",
      domain: "unknown",
      phone_format: "unknown",
      store_signs: detectSigns(lowerDesc)
    },
    architecture: {
      style: detectArchitecture(lowerDesc),
      colors: detectColors(lowerDesc),
      roof_type: detectRoofType(lowerDesc),
      density: detectDensity(lowerDesc)
    },
    vehicles: {
      brands: detectVehicles(lowerDesc),
      markings: "unknown",
      bus_text: "unknown", 
      parking_style: "unknown"
    },
    cultural_indicators: {
      religion: "unknown",
      flag: "unknown",
      murals: "unknown"
    },
    environment: {
      vegetation: detectVegetation(lowerDesc),
      terrain: detectTerrain(lowerDesc), 
      climate_hint: detectClimate(lowerDesc),
      coast_proximity: detectCoast(lowerDesc),
      altitude: detectAltitude(lowerDesc)
    },
    meta: {
      police_presence: "false",
      escort_vehicle: "false", 
      camera_shadow: "unknown",
      unique_clues: extractUniqueClues(description)
    },
    original_description: description
  };
  
  return features;
}

// Helper functions for feature detection
function detectSurface(desc) {
  if (desc.includes('asphalt') || desc.includes('paved')) return "asphalt";
  if (desc.includes('gravel')) return "gravel";
  if (desc.includes('dirt') || desc.includes('unpaved')) return "dirt";
  if (desc.includes('cobblestone')) return "cobblestone";
  return "unknown";
}

function detectCurvature(desc) {
  if (desc.includes('straight')) return "straight";
  if (desc.includes('winding') || desc.includes('curved')) return "winding";
  if (desc.includes('switchback')) return "switchbacks";
  return "unknown";
}

function detectElevation(desc) {
  if (desc.includes('flat') || desc.includes('level')) return "flat";
  if (desc.includes('hill') || desc.includes('slope')) return "hilly";
  if (desc.includes('mountain')) return "mountainous";
  return "unknown";
}

function detectPoles(desc) {
  if (desc.includes('wooden pole') || desc.includes('wood pole')) return "wooden poles";
  if (desc.includes('concrete pole')) return "concrete poles";
  if (desc.includes('metal pole')) return "metal poles";
  return "unknown";
}

function detectLanguage(desc) {
  if (desc.includes('english') || desc.includes('latin alphabet')) return "English";
  if (desc.includes('spanish')) return "Spanish";
  if (desc.includes('french')) return "French";
  if (desc.includes('german')) return "German";
  if (desc.includes('cyrillic')) return "Cyrillic";
  if (desc.includes('arabic')) return "Arabic";
  if (desc.includes('chinese') || desc.includes('mandarin')) return "Chinese";
  return "unknown";
}

function detectSigns(desc) {
  if (desc.includes('stop sign')) return "stop signs";
  if (desc.includes('yield sign')) return "yield signs";
  if (desc.includes('speed limit')) return "speed limit signs";
  if (desc.includes('street sign')) return "street signs";
  return "unknown";
}

function detectArchitecture(desc) {
  if (desc.includes('colonial')) return "colonial";
  if (desc.includes('modern')) return "modern";
  if (desc.includes('traditional')) return "traditional";
  if (desc.includes('soviet') || desc.includes('communist')) return "soviet/eastern bloc";
  if (desc.includes('mediterranean')) return "mediterranean";
  if (desc.includes('scandinavian')) return "scandinavian";
  return "unknown";
}

function detectColors(desc) {
  const colors = [];
  if (desc.includes('red')) colors.push('red');
  if (desc.includes('white')) colors.push('white');
  if (desc.includes('blue')) colors.push('blue');
  if (desc.includes('yellow')) colors.push('yellow');
  if (desc.includes('green')) colors.push('green');
  return colors.length > 0 ? colors.join(', ') : "unknown";
}

function detectRoofType(desc) {
  if (desc.includes('flat roof')) return "flat";
  if (desc.includes('pitched roof') || desc.includes('sloped roof')) return "pitched";
  if (desc.includes('tile roof') || desc.includes('tiled roof')) return "tiled";
  if (desc.includes('metal roof')) return "metal";
  return "unknown";
}

function detectDensity(desc) {
  if (desc.includes('urban') || desc.includes('city') || desc.includes('downtown')) return "urban";
  if (desc.includes('suburban') || desc.includes('residential')) return "suburban";
  if (desc.includes('rural') || desc.includes('countryside') || desc.includes('village')) return "rural";
  return "unknown";
}

function detectVehicles(desc) {
  const brands = [];
  if (desc.includes('toyota')) brands.push('Toyota');
  if (desc.includes('honda')) brands.push('Honda');
  if (desc.includes('ford')) brands.push('Ford');
  if (desc.includes('volkswagen') || desc.includes('vw')) brands.push('Volkswagen');
  if (desc.includes('mercedes')) brands.push('Mercedes');
  if (desc.includes('bmw')) brands.push('BMW');
  return brands.length > 0 ? brands.join(', ') : "unknown";
}

function detectVegetation(desc) {
  if (desc.includes('tropical') || desc.includes('palm')) return "tropical";
  if (desc.includes('temperate') || desc.includes('deciduous')) return "temperate";
  if (desc.includes('coniferous') || desc.includes('pine') || desc.includes('evergreen')) return "coniferous";
  if (desc.includes('desert') || desc.includes('arid')) return "arid/desert";
  if (desc.includes('mediterranean')) return "mediterranean";
  if (desc.includes('grassland') || desc.includes('prairie')) return "grassland";
  return "unknown";
}

function detectTerrain(desc) {
  if (desc.includes('flat') || desc.includes('plains')) return "flat";
  if (desc.includes('hilly') || desc.includes('hills')) return "hilly";
  if (desc.includes('mountain') || desc.includes('mountainous')) return "mountains";
  if (desc.includes('valley')) return "valley";
  return "unknown";
}

function detectClimate(desc) {
  if (desc.includes('tropical') || desc.includes('humid')) return "tropical";
  if (desc.includes('arid') || desc.includes('dry') || desc.includes('desert')) return "arid";
  if (desc.includes('temperate')) return "temperate";
  if (desc.includes('cold') || desc.includes('snow')) return "cold";
  if (desc.includes('mediterranean')) return "mediterranean";
  return "unknown";
}

function detectCoast(desc) {
  if (desc.includes('ocean') || desc.includes('sea') || desc.includes('coast') || desc.includes('beach')) return "true";
  return "false";
}

function detectAltitude(desc) {
  if (desc.includes('high altitude') || desc.includes('mountain')) return "highland";
  if (desc.includes('sea level') || desc.includes('lowland')) return "lowland";
  return "unknown";
}

function extractUniqueClues(desc) {
  const clues = [];
  if (desc.includes('google street view')) clues.push('Street View imagery');
  if (desc.includes('license plate')) clues.push('Visible license plates');
  if (desc.includes('street sign')) clues.push('Street signage visible');
  if (desc.includes('flag')) clues.push('National flag visible');
  return clues.length > 0 ? clues.join(', ') : desc.substring(0, 100) + '...';
}

// Nemotron service function with comprehensive GeoGuessr prompt
async function predictCountryWithNemotron(features) {
  try {
    console.log('📤 Sending request to Nemotron...');
    
    const systemPrompt = `You are a country classification model trained on expert GeoGuessr strategies.

Given the following image-derived input features, predict the most likely country:

TASK: Predict the most likely country and rationale based on the above inputs.

OUTPUT:
{
  "predicted_country": "<country name>",
  "confidence": "<low|medium|high|very high>",
  "rationale": [
    "<reason 1>",
    "<reason 2>",
    "...etc"
  ]
}`;

    const userPrompt = `INPUT:
${JSON.stringify(features, null, 2)}

TASK: Predict the most likely country and rationale based on the above inputs.

You MUST analyze the comprehensive feature set and make a specific prediction. Pay special attention to:
- Architecture styles (colonial, modern, traditional, soviet, mediterranean, scandinavian)
- Environment indicators (vegetation, climate, terrain, coast proximity)
- Infrastructure quality (road surface, density, poles)
- Cultural markers (language, signs, vehicles)

Do not default to common countries like US unless features specifically support it.

OUTPUT (JSON only):`;

    const payload = {
      model: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
      messages: [
        {
          role: "system",
          content: systemPrompt
        },
        {
          role: "user",
          content: userPrompt
        }
      ],
      temperature: 0.4,
      max_tokens: 500,
      top_p: 0.9
    };

    const response = await axios.post(NVIDIA_API_URL, payload, {
      headers: {
        'Authorization': `Bearer ${NVIDIA_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    console.log('📥 Nemotron response received:', response.status);

    if (response.status === 200) {
      const content = response.data.choices[0].message.content;
      console.log('🔍 Nemotron content:', content);
      
      // Extract JSON from response
      const startIdx = content.indexOf('{');
      const endIdx = content.lastIndexOf('}') + 1;
      
      if (startIdx !== -1 && endIdx > startIdx) {
        const jsonStr = content.substring(startIdx, endIdx);
        const prediction = JSON.parse(jsonStr);
        
        // Ensure we have a valid country (not "unknown" or similar)
        if (!prediction.predicted_country || 
            prediction.predicted_country.toLowerCase().includes('unknown') ||
            prediction.predicted_country.toLowerCase().includes('unclear') ||
            prediction.predicted_country.toLowerCase().includes('uncertain')) {
          // Force a prediction based on comprehensive features
          prediction.predicted_country = generateFallbackCountryPrediction(features);
          prediction.confidence = "low";
          prediction.rationale = [
            "Limited visual information available",
            "Making educated guess based on detected patterns",
            "Used fallback analysis of available features"
          ];
        }
        
        // Convert confidence to score if needed
        const confidenceMap = {
          'low': 35,
          'medium': 65,
          'high': 85,
          'very high': 95
        };
        prediction.confidence_score = confidenceMap[prediction.confidence?.toLowerCase()] || 50;
        
        // Create features string for display - extract key detected features
        const keyFeatures = [];
        if (features.road?.surface !== "unknown") keyFeatures.push(`Road: ${features.road.surface}`);
        if (features.architecture?.style !== "unknown") keyFeatures.push(`Architecture: ${features.architecture.style}`);
        if (features.environment?.vegetation !== "unknown") keyFeatures.push(`Vegetation: ${features.environment.vegetation}`);
        if (features.environment?.climate_hint !== "unknown") keyFeatures.push(`Climate: ${features.environment.climate_hint}`);
        if (features.text_features?.language !== "unknown") keyFeatures.push(`Language: ${features.text_features.language}`);
        if (features.vehicles?.brands !== "unknown") keyFeatures.push(`Vehicles: ${features.vehicles.brands}`);
        if (features.architecture?.density !== "unknown") keyFeatures.push(`Density: ${features.architecture.density}`);
        
        prediction.features = keyFeatures.length > 0 
          ? keyFeatures.join(', ') 
          : 'Visual analysis of image features';
        
        return prediction;
      } else {
        throw new Error('No valid JSON found in Nemotron response');
      }
    } else {
      throw new Error(`Nemotron API error: ${response.status}`);
    }
  } catch (error) {
    console.error('❌ Nemotron prediction error:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return getFallbackPrediction(features);
  }
}

// Generate a fallback country prediction based on GeoGuessr patterns
function generateFallbackCountryPrediction(features) {
  console.log('🔧 Generating fallback prediction from features:', JSON.stringify(features, null, 2));
  
  // Use comprehensive GeoGuessr logic - prioritize unique indicators
  
  // Language is often the strongest indicator
  if (features.text_features?.language !== "unknown") {
    const lang = features.text_features.language.toLowerCase();
    if (lang === "spanish") return "Spain";
    if (lang === "french") return "France"; 
    if (lang === "german") return "Germany";
    if (lang === "cyrillic") return "Russia";
    if (lang === "chinese") return "China";
    if (lang === "arabic") return "Morocco";
  }
  
  // Climate + vegetation combinations (strong regional indicators)
  const climate = features.environment?.climate_hint;
  const vegetation = features.environment?.vegetation;
  const terrain = features.environment?.terrain;
  const coast = features.environment?.coast_proximity;
  
  if (climate === "tropical") {
    if (coast === "true") return "Thailand";  // Tropical coastal
    if (vegetation === "tropical") return "Indonesia";
    return "Malaysia";
  }
  
  if (climate === "arid" || vegetation === "arid/desert") {
    if (terrain === "mountains") return "Chile";
    if (features.architecture?.style === "modern") return "Australia";
    return "Morocco";
  }
  
  if (vegetation === "coniferous") {
    if (terrain === "mountains") return "Norway";
    if (climate === "cold") return "Finland";
    return "Canada";
  }
  
  if (vegetation === "mediterranean") {
    if (features.architecture?.style === "traditional") return "Greece";
    return "Italy";
  }
  
  // Architecture style indicators
  const archStyle = features.architecture?.style;
  if (archStyle === "soviet/eastern bloc") return "Poland";
  if (archStyle === "colonial") return "Mexico";
  if (archStyle === "scandinavian") return "Sweden";
  if (archStyle === "mediterranean") return "Spain";
  
  // Infrastructure quality + density combinations
  const roadSurface = features.road?.surface;
  const density = features.architecture?.density;
  
  if (roadSurface === "dirt" && density === "rural") return "Kenya";
  if (roadSurface === "cobblestone") return "Netherlands";
  if (density === "urban" && archStyle === "modern") return "Japan";
  
  // Terrain-based fallbacks
  if (terrain === "mountains") {
    if (vegetation === "temperate") return "Switzerland";
    if (climate === "cold") return "Austria";
    return "Peru";
  }
  
  if (terrain === "flat") {
    if (vegetation === "grassland") return "Argentina";
    if (density === "rural") return "Denmark";
    return "Netherlands";
  }
  
  // Coastal areas
  if (coast === "true") {
    if (climate === "temperate") return "Portugal";
    if (vegetation === "mediterranean") return "Croatia";
    return "New Zealand";
  }
  
  // Vehicle brand indicators (less reliable but useful)
  const vehicles = features.vehicles?.brands;
  if (vehicles && vehicles !== "unknown") {
    if (vehicles.includes("Toyota") && climate === "tropical") return "Philippines";
    if (vehicles.includes("Volkswagen")) return "Germany";
  }
  
  // Default fallbacks - avoid US bias, use diverse options
  const fallbackCountries = [
    "United Kingdom", "France", "Germany", "Italy", "Spain", 
    "Australia", "Canada", "Brazil", "Argentina", "South Africa",
    "Sweden", "Norway", "Poland", "Czech Republic", "Japan"
  ];
  
  // Pick based on simple hash of features to be consistent
  const featureString = JSON.stringify(features);
  const hash = featureString.length % fallbackCountries.length;
  
  console.log('🎯 Selected fallback country:', fallbackCountries[hash]);
  return fallbackCountries[hash];
}

// Helper functions
function getFallbackFeatures() {
  return {
    sun_dir: "unknown",
    cam_gen: "unknown",
    drive_side: "unknown",
    road: {
      lines: "unknown",
      surface: "unknown", 
      shoulder: "unknown",
      median: "unknown",
      curvature: "unknown",
      elevation: "unknown"
    },
    bollards: "unknown",
    poles: "unknown",
    guardrails: "unknown",
    signs: {
      lang: "unknown",
      shapes: "unknown",
      units: "unknown", 
      mounts: "unknown"
    },
    license_plate: {
      front: "unknown",
      rear: "unknown",
      blur_status: "unknown",
      country_code: "unknown"
    },
    text_features: {
      language: "unknown",
      toponyms: "unknown",
      domain: "unknown",
      phone_format: "unknown",
      store_signs: "unknown"
    },
    architecture: {
      style: "unknown",
      colors: "unknown", 
      roof_type: "unknown",
      density: "unknown"
    },
    vehicles: {
      brands: "unknown",
      markings: "unknown",
      bus_text: "unknown",
      parking_style: "unknown"
    },
    cultural_indicators: {
      religion: "unknown",
      flag: "unknown", 
      murals: "unknown"
    },
    environment: {
      vegetation: "unknown",
      terrain: "unknown",
      climate_hint: "unknown",
      coast_proximity: "false",
      altitude: "unknown"
    },
    meta: {
      police_presence: "false",
      escort_vehicle: "false",
      camera_shadow: "unknown",
      unique_clues: "Unable to analyze image with AI models"
    },
    original_description: "AI model analysis failed"
  };
}

function getFallbackPrediction(features) {
  return {
    predicted_country: generateFallbackCountryPrediction(features),
    confidence: "low",
    confidence_score: 25,
    rationale: [
      "AI model analysis failed, using pattern-based prediction",
      "Prediction based on common geographic and architectural patterns", 
      "Limited visual information available for detailed analysis"
    ],
    features: 'Fallback analysis using geographic patterns'
  };
}

// Main classification endpoint
app.post('/classify-country/', upload.single('file'), async (req, res) => {
  try {
    console.log('🔍 Starting country classification...');
    
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No image file provided' 
      });
    }

    console.log(`📷 Image received: ${req.file.mimetype}, ${(req.file.size / 1024).toFixed(1)}KB`);

    // Process image with Sharp (optional optimization)
    let imageBuffer = req.file.buffer;
    try {
      // Optimize image size if too large
      const metadata = await sharp(imageBuffer).metadata();
      console.log(`🖼️ Image dimensions: ${metadata.width}x${metadata.height}`);
      
      if (metadata.width > 1024 || metadata.height > 1024) {
        console.log('🔧 Resizing image...');
        imageBuffer = await sharp(imageBuffer)
          .resize(1024, 1024, { fit: 'inside', withoutEnlargement: true })
          .jpeg({ quality: 85 })
          .toBuffer();
      }
    } catch (sharpError) {
      console.log('⚠️ Sharp optimization failed, using original image');
    }

    // Stage 1: Extract features with Florence-2
    console.log('🔍 Extracting features with Florence-2...');
    const features = await extractFeaturesWithFlorence(imageBuffer);
    
    // Stage 2: Predict country with Nemotron
    console.log('🌍 Predicting country with Nemotron...');
    const prediction = await predictCountryWithNemotron(features);
    
    // Format response for frontend
    const response = {
      features: prediction.features || 'Visual analysis',
      country: prediction.predicted_country || 'Unknown',
      confidence: prediction.confidence_score || 25,
      explanation: Array.isArray(prediction.rationale) 
        ? prediction.rationale.join('. ') 
        : prediction.rationale || 'No explanation available'
    };
    
    console.log('✅ Classification completed:', response.country, `(${response.confidence}%)`);
    res.json(response);
    
  } catch (error) {
    console.error('❌ Classification error:', error.message);
    res.status(500).json({
      error: 'Classification failed',
      details: error.message
    });
  }
});

// Alternative endpoint without trailing slash
app.post('/classify-country', upload.single('file'), async (req, res) => {
  req.url = '/classify-country/';
  return app._router.handle(req, res);
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    message: 'Country Classifier API is running',
    timestamp: new Date().toISOString(),
    api_keys: {
      florence: FLORENCE_API_KEY ? 'Present' : 'Missing',
      nvidia: NVIDIA_API_KEY ? 'Present' : 'Missing'
    }
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({ 
    message: 'Country Classifier API',
    version: '1.0.0',
    endpoints: {
      classify: 'POST /classify-country/',
      health: 'GET /health'
    }
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large (max 10MB)' });
    }
  }
  
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log('\n🚀 Country Classifier API Starting...');
  console.log(`📍 Server running on: http://localhost:${PORT}`);
  console.log(`🔍 Classification endpoint: http://localhost:${PORT}/classify-country/`);
  console.log(`💚 Health check: http://localhost:${PORT}/health`);
  console.log('');
  console.log('Pipeline: Image → Florence-2 → Nemotron → Country Prediction');
  console.log('');
  
  // Final API Key Check
  console.log('🔑 FINAL API KEY STATUS:');
  console.log(`   Florence-2: ${FLORENCE_API_KEY ? '✅ Present (' + FLORENCE_API_KEY.length + ' chars)' : '❌ Missing'}`);
  console.log(`   NVIDIA: ${NVIDIA_API_KEY ? '✅ Present (' + NVIDIA_API_KEY.length + ' chars)' : '❌ Missing'}`);
  
  console.log('\n✅ Ready to classify countries! 🌍');
});