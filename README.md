'''mermaid
graph TD
    subgraph External ["External Environment"]
        User((Patient / User))
        GoogleFit[Google Fit API]
        Jenkins[CI/CD Trigger / GitHub Actions]
    end

    subgraph DockerHost ["Docker Engine (docker-compose)"]
        
        subgraph Volumes ["Persistent Docker Volumes"]
            PGData[(pg_data_volume <br/> PostgreSQL Storage)]
            ModelVol[("model_volume <br/> (Shared inference.pkl)")]
        end
        
        subgraph Containers ["Isolated Microservices (lif-network)"]
            Frontend[lif-frontend <br/> Streamlit Dashboard <br/> Exposed Port: 8501, 8080]
            Database[(lif-database <br/> PostgreSQL 15 <br/> Internal Port: 5432)]
            Worker[lif-ml-worker <br/> Headless Python Environment]
        end
        
    end

    %% External to Container Connections
    User -- "Access UI (Port 8501)" --> Frontend
    Frontend -- "Request OAuth 2.0" --> GoogleFit
    GoogleFit -- "Return Telemetry (Steps/Sleep)" --> Frontend
    Jenkins -- "Trigger Model Retraining" --> Worker
    
    %% Internal Container Interactions
    Frontend -- "Save Logs & Dictionary Maps (UPSERT)" --> Database
    Worker -- "Fetch 217k+ Rows for MLE (SQL)" --> Database
    
    %% Volume Persistence
    Database === "Mounts & Persists Data" === PGData
    Worker -- "Serializes new .pkl" --> ModelVol
    ModelVol -. "Hot-reloads .pkl for Inference" .-> Frontend
    
    %% Styling
    classDef container fill:#1f618d,stroke:#154360,stroke-width:2px,color:#fff;
    classDef db fill:#1e8449,stroke:#145a32,stroke-width:2px,color:#fff;
    classDef volume fill:#d68910,stroke:#935116,stroke-width:2px,color:#fff;
    classDef external fill:#7d3c98,stroke:#4a235a,stroke-width:2px,color:#fff;
    
    class Frontend,Worker container;
    class Database db;
    class PGData,ModelVol volume;
    class GoogleFit,Jenkins external;
'''
