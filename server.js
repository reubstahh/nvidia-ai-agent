require('dotenv').config();

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const axios = require('axios');
const sharp = require('sharp');
const FormData = require('form-data');

const app = express();
const PORT = 8000;

// API Keys
const FLORENCE_API_KEY = process.env.FLORENCE_API_KEY;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

// API URLs
const FLORENCE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large";
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

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

// GeoGuessr feature extraction prompt for Florence-2
const GEOGUESSR_PROMPT = `
You are a GeoGuessr expert analyzing this image for country identification clues.

Extract and categorize ALL visible details into this exact JSON structure:

{
  "sun_dir": "<north|south|overhead|unknown>",
  "cam_gen": "<Gen 1|Gen 2|Gen 3|Cam bars|No car>",
  "drive_side": "<left|right>",
  "road": {
    "lines": "<e.g. dashed white center, yellow outer>",
    "surface": "<asphalt|gravel|cobblestone|dirt>",
    "shoulder": "<gravel|dirt|paved|none>",
    "median": "<e.g. concrete barrier>",
    "curvature": "<straight|winding|switchbacks>",
    "elevation": "<flat|hilly|mountainous>"
  },
  "bollards": "<style and color>",
  "poles": "<e.g. wood, concrete, shape>",
  "guardrails": "<type or absence>",
  "signs": {
    "lang": "<language(s)>",
    "shapes": "<common shapes and colors>",
    "units": "<km/h|mph>",
    "mounts": "<mount style>"
  },
  "license_plate": {
    "front": "<color + region marker>",
    "rear": "<color + shape>",
    "blur_status": "<blurred|unblurred>",
    "country_code": "<e.g. CL, ZA>"
  },
  "text_features": {
    "language": "<detected text language(s)>",
    "toponyms": "<place names>",
    "domain": "<e.g. .cl, .jp>",
    "phone_format": "<e.g. +56>",
    "store_signs": "<examples of signage text>"
  },
  "architecture": {
    "style": "<building type>",
    "colors": "<dominant palette>",
    "roof_type": "<flat|pitched|tiled>",
    "density": "<urban|rural|suburban>"
  },
  "vehicles": {
    "brands": "<car brands>",
    "markings": "<business or location clues>",
    "bus_text": "<company name if any>",
    "parking_style": "<parallel|angle|head-in>"
  },
  "cultural_indicators": {
    "religion": "<visible symbols>",
    "flag": "<country flags seen>",
    "murals": "<art motifs>"
  },
  "environment": {
    "vegetation": "<climate zone>",
    "terrain": "<flat|hilly|mountains>",
    "climate_hint": "<e.g. arid, tropical>",
    "coast_proximity": "<true|false>",
    "altitude": "<lowland|highland>"
  },
  "meta": {
    "police_presence": "<true|false>",
    "escort_vehicle": "<true|false>",
    "camera_shadow": "<visible|not visible>",
    "unique_clues": "<summary>"
  }
}

Respond with ONLY the JSON structure filled with your observations. Use "unknown" or "none" for unclear items.
`;

// Florence-2 service function
async function extractFeaturesWithFlorence(imageBuffer) {
  try {
    // Convert image to base64
    const base64Image = imageBuffer.toString('base64');
    
    const payload = {
      inputs: {
        image: base64Image,
        text: GEOGUESSR_PROMPT
      },
      parameters: {
        max_new_tokens: 1000,
        temperature: 0.1,
        do_sample: false
      }
    };

    const response = await axios.post(FLORENCE_API_URL, payload, {
      headers: {
        'Authorization': `Bearer ${FLORENCE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 60000
    });

    if (response.status === 200) {
      let generatedText = '';
      
      if (Array.isArray(response.data) && response.data.length > 0) {
        generatedText = response.data[0].generated_text || '';
      } else {
        generatedText = response.data.generated_text || '';
      }

      // Extract JSON from response
      const startIdx = generatedText.indexOf('{');
      const endIdx = generatedText.lastIndexOf('}') + 1;
      
      if (startIdx !== -1 && endIdx > startIdx) {
        const jsonStr = generatedText.substring(startIdx, endIdx);
        return JSON.parse(jsonStr);
      } else {
        throw new Error('No valid JSON found in Florence-2 response');
      }
    } else {
      throw new Error(`Florence-2 API error: ${response.status}`);
    }
  } catch (error) {
    console.error('Florence-2 extraction error:', error.message);
    return getFallbackFeatures();
  }
}

// Nemotron service function
async function predictCountryWithNemotron(features) {
  try {
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
      temperature: 0.1,
      max_tokens: 1000,
      top_p: 0.9
    };

    const response = await axios.post(NVIDIA_API_URL, payload, {
      headers: {
        'Authorization': `Bearer ${NVIDIA_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 60000
    });

    if (response.status === 200) {
      const content = response.data.choices[0].message.content;
      
      // Extract JSON from response
      const startIdx = content.indexOf('{');
      const endIdx = content.lastIndexOf('}') + 1;
      
      if (startIdx !== -1 && endIdx > startIdx) {
        const jsonStr = content.substring(startIdx, endIdx);
        const prediction = JSON.parse(jsonStr);
        
        // Convert confidence to score if needed
        if (!prediction.confidence_score) {
          const confidenceMap = {
            'low': 25,
            'medium': 50,
            'high': 75,
            'very high': 90
          };
          prediction.confidence_score = confidenceMap[prediction.confidence?.toLowerCase()] || 50;
        }
        
        // Ensure features string exists
        if (!prediction.features) {
          prediction.features = extractKeyFeatures(features);
        }
        
        return prediction;
      } else {
        throw new Error('No valid JSON found in Nemotron response');
      }
    } else {
      throw new Error(`Nemotron API error: ${response.status}`);
    }
  } catch (error) {
    console.error('Nemotron prediction error:', error.message);
    return getFallbackPrediction(features);
  }
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
      coast_proximity: "unknown",
      altitude: "unknown"
    },
    meta: {
      police_presence: "unknown",
      escort_vehicle: "unknown",
      camera_shadow: "unknown",
      unique_clues: "Image analysis failed, using fallback data"
    }
  };
}

function extractKeyFeatures(features) {
  const keyFeatures = [];
  
  if (features.license_plate?.country_code !== "unknown") {
    keyFeatures.push(`License plate: ${features.license_plate.country_code}`);
  }
  
  if (features.text_features?.language !== "unknown") {
    keyFeatures.push(`Language: ${features.text_features.language}`);
  }
  
  if (features.signs?.lang !== "unknown") {
    keyFeatures.push(`Sign language: ${features.signs.lang}`);
  }
  
  if (features.drive_side !== "unknown") {
    keyFeatures.push(`Drive side: ${features.drive_side}`);
  }
  
  if (features.environment?.vegetation !== "unknown") {
    keyFeatures.push(`Vegetation: ${features.environment.vegetation}`);
  }
  
  if (features.architecture?.style !== "unknown") {
    keyFeatures.push(`Architecture: ${features.architecture.style}`);
  }
  
  return keyFeatures.length > 0 ? keyFeatures.join(', ') : 'General visual analysis';
}

function getFallbackPrediction(features) {
  return {
    predicted_country: "Unknown",
    confidence: "low",
    confidence_score: 25,
    rationale: [
      "API analysis failed",
      "Using fallback prediction system",
      "Please try again with a different image"
    ],
    features: extractKeyFeatures(features)
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

    // Process image with Sharp (optional optimization)
    let imageBuffer = req.file.buffer;
    try {
      // Optimize image size if too large
      const metadata = await sharp(imageBuffer).metadata();
      if (metadata.width > 1024 || metadata.height > 1024) {
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
      features: prediction.features || '',
      country: prediction.predicted_country || 'Unknown',
      confidence: prediction.confidence_score || 0,
      explanation: Array.isArray(prediction.rationale) 
        ? prediction.rationale.join('. ') 
        : prediction.rationale || 'No explanation available'
    };
    
    console.log('✅ Classification completed:', response.country);
    res.json(response);
    
  } catch (error) {
    console.error('❌ Classification error:', error.message);
    res.status(500).json({
      error: 'Classification failed',
      details: error.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    message: 'Country Classifier API is running' 
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
  console.log('🚀 Country Classifier API Starting...');
  console.log(`📍 Server running on: http://localhost:${PORT}`);
  console.log(`🔍 Classification endpoint: http://localhost:${PORT}/classify-country/`);
  console.log(`💚 Health check: http://localhost:${PORT}/health`);
  console.log('');
  console.log('Pipeline: Image → Florence-2 → Nemotron → Country Prediction');
  console.log('Ready to classify countries! 🌍');
});