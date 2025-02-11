/* eslint-disable */
import { useEffect, useState, useContext } from 'react';

// material-ui
import { Grid, Box, Typography, Container, Paper, Divider, Button } from '@mui/material';
import { makeStyles } from '@mui/styles';
import { gridSpacing } from 'store/constant';
import OptionsChoice from 'ui-component/OptionsChoice';
import { ParametersContext } from 'context';

import OutlinedInput from '@mui/material/OutlinedInput';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import ListItemText from '@mui/material/ListItemText';
import Select from '@mui/material/Select';
import Checkbox from '@mui/material/Checkbox';
import Alert from '@mui/material/Alert';

// Import the images
import LargeDataCollectionImage from '../../../assets/images/B1._Large_Data_Included.png';
import CuratedDataImage from 'assets/images/B2._Data_Preprocessing.png';
import MachineLearningImage from 'assets/images/B3._Advanced_AI-assisted_QSAR_Models.png';
import ReliablePredictionsImage from 'assets/images/B4._User-Friendly_Interface.png';

import axios from 'axios';

import coverimage from 'assets/images/muti-task_interface_homepage_PW2.png';
import coverimage1 from 'assets/images/phhp.png';
import { color } from 'framer-motion';
import { Stack } from '@mui/system';
import options from './options';
// ==============================|| DEFAULT DASHBOARD ||============================== //

const useStyles = makeStyles((theme) => ({
  topImage: {
    width: '60%',
    height: 'auto',
    display: 'block',
    marginLeft: 'auto',
    marginRight: 'auto',
  },
  curvedContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: '15px',
    padding: '40px',
    margin: '20px auto',
    width: '80%',
    boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
    border: '2px solid blue', // Added border property for blue border
  },
  iconSection: {
    textAlign: 'center',
    marginTop: '20px',
  },
  icon: {
    fontSize: '50px',
    color: theme.palette.primary.main,
    width: 'auto', // Set the desired width
    height: '156px', // Set the desired height
    objectFit: 'contain', // Ensure the image fits within the specified dimensions
    marginBottom: theme.spacing(2),
  },
  iconText: {
    marginTop: '10px',
    fontWeight: 'bold',
  },
  divider: {
    height: '100%',
    backgroundColor: theme.palette.primary.main,
  },
  iconWrapper: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
}));

const VariblesMap = {
  "CAS": "CAS",
  "Name": "Name",
  "SMILES": "SMILES"
}

const names = [
  "CAS",
  "Name",
  "SMILES"
];

const tox21 = [ 
  "Include",
  "Do not include"
 ]

const toxicityTypes = [
  'Cardiotoxicity',
  'Developmental Toxicity',
  'Hepatotoxicity',
  'Nephrotoxicity',
  'Neurotoxicity',
  'Reproductive Toxicity'
];

const threshold = 0.5;

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

const Dashboard = () => {
  const classes = useStyles();
  const { url } = useContext(ParametersContext);

  const [personName, setPersonName] = useState("Name");
  const [include, setInclude] = useState("Include");
  const [data, setdata] = useState("");
  const [response, setResponse] = useState(null);

  const [fileName, setFileName] = useState(null); // To store the file name
  const [fileContent, setFileContent] = useState(''); // To store the file content
  const fileContentLst = fileContent.split(', ')
  // const options = optionsTm;
  

  const handleChange = (event) => {
    const {
      target: { value },
    } = event;
    setPersonName(value);
  };

  const handleChangeInclude = (event) => {
    const {
      target: { value },
    } = event;
    setInclude(value);
  };

  const handleData = (event) => {
    const {
      target: { value },
    } = event;
    setdata(value);
  };

  const handleClear = () => {
    setResponse(null);
    setdata("");
    setFileContent("");
    setFileName(null);
    setPersonName("Name");
    setInclude("Include");
    console.log("Inputs and responses cleared");
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
  
    if (file) {
      setFileName(file.name); // Set the file name
  
      const reader = new FileReader();
      reader.onload = (event) => {
        const fileContent = event.target.result; // Get file content
        const formattedContent = fileContent.replace(/[\r\n]+/g, ', ');
        setFileContent(formattedContent); // Save formatted content to state
        console.log('Formatted File Content:', formattedContent);
      };
  
      reader.onerror = () => {
        console.error('Error reading file');
      };
  
      reader.readAsText(file); // Read the file as text
    }
  };

  const handleExecute = async () => {
    let postData=null;
    try {
      if(fileContent!==''){
        postData = { caslist: fileContent, toxdata: include.toLowerCase()};
      }
      else if(data!=="") {
        postData = { caslist: options['SMILES'][data], toxdata: include.toLowerCase()};
      }
      if(postData!==null){
        console.log('postData',postData)
        const response = await axios.post(`${url}/pfas/prediction`, postData);
        console.log("response.data", response.data.y_hatRem)
        setResponse(response.data.y_hatRem);
      }
    } catch (error) {
      console.error("Error executing the request", error);
    }
  };

  return (
    <>
      <Grid container spacing={gridSpacing} justifyContent="center" alignItems="center" direction="column">
        <Grid item xs={12}>
          <img src={coverimage} alt="Top Image" className={classes.topImage} /> {/* Update with your image path */}
          <Paper className={classes.curvedContainer}>
  <Grid
    container
    direction="column"
    alignItems="center"
    justifyContent="center"
    spacing={3}
    style={{ textAlign: 'center' }}
  >
    <Grid item xs={12}>
      <Typography variant="h5" gutterBottom style={{ fontWeight: 'bold' }}>
        Introduction
      </Typography>
    </Grid>
    <Grid item xs={12} style={{ width: '60%', margin: '0 auto' }}>
      <Typography variant="body1" align="justify" style={{ lineHeight: 1.8 }}>
      This AI-assisted QSAR model interface provides a robust tool for simultaneously predicting multi-organ toxicities in humans. Six endpoints have been selected to represent various human organ-level adverse outcomes, including cardiotoxicity, developmental toxicity, hepatotoxicity, nephrotoxicity, neurotoxicity, and reproductive toxicity. These predictions play a vital role in predicting multiple in vivo organ toxicities, enhancing success rates in the early stage of drug development, and mitigating drug failure risks.

      First, choose whether to include Tox21 in vitro high-throughput screening bioactivity assay data. If you opt to include Tox21 data, you can easily select the name, CAS, or SMILES of chemicals to receive yes or no prediction results for human health outcomes from our robust AI-QSAR model. If you choose not to include Tox21 data, you can upload your own CSV file containing the CAS number of chemicals or select the name, CAS, or SMILES of chemicals to obtain prediction results.
      </Typography>
    </Grid>
    <Grid container direction="row" justifyContent="center" alignItems="center" spacing={3} marginTop={1}>
      <Grid item>
        <FormControl sx={{ m: 1, width: 300 }}>
          <InputLabel id="tox21-label">Tox21 data</InputLabel>
          <Select
            labelId="tox21-label"
            id="tox21-select"
            value={include}
            onChange={handleChangeInclude}
            input={<OutlinedInput label="Tag" />}
            MenuProps={MenuProps}
          >
            {tox21.map((name) => (
              <MenuItem key={name} value={name}>
                {name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item>
        <FormControl sx={{ m: 1, width: 300 }}>
          <InputLabel id="input-type-label">Input Type</InputLabel>
          <Select
            labelId="input-type-label"
            id="input-type-select"
            value={personName}
            onChange={handleChange}
            input={<OutlinedInput label="Tag" />}
            MenuProps={MenuProps}
          >
            {names.map((name) => (
              <MenuItem key={name} value={name}>
                {name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item>
        <FormControl sx={{ m: 1, width: 300 }}>
          <InputLabel id="variables-map-label">{VariblesMap[personName]}</InputLabel>
          <Select
            labelId="variables-map-label"
            id="variables-map-select"
            value={data}
            onChange={handleData}
            input={<OutlinedInput label="Tag" />}
            MenuProps={MenuProps}
          >
            {options[personName].map((name, i) => (
              <MenuItem key={i} value={i}>
                {name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
    </Grid>
    {include === 'Do not include' && (
      <Grid item xs={12}>
        <Box
          style={{
            border: '2px dashed #ccc',
            padding: '50px',
            textAlign: 'center',
            cursor: 'pointer',
            minHeight: '200px',
            marginTop: '20px',
          }}
        >
          <Typography variant="h4" sx={{ mb: 2 }}>
            <strong>File Input</strong>
          </Typography>
          <Typography variant="body2">
            Click to upload
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Please upload a CSV file containing columns: &apos;<strong>SMILES</strong>&apos;
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Maximum Limit on Number of rows: <strong>56</strong>
          </Typography>

          <Box marginTop="16px">
            <Button variant="contained" component="label">
              Choose Files
              <input
                type="file"
                accept=".csv" // Specify only CSV files
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
              </Button>
            </Box>

            {fileName && (
              <Box marginTop="16px">
                <Typography variant="subtitle1">
                  Uploaded File: <strong>{fileName}</strong>
                </Typography>
              </Box>
            )}

          </Box>
        </Grid>
      )}
      <Grid item xs={12}>
      <Box 
    display="flex" 
    justifyContent="center" 
    gap="16px" // Adjust the gap size as needed
  >
    <Button
      variant="contained"
      color="primary"
      onClick={handleExecute}
      style={{
        padding: '10px 20px',
        fontSize: '16px',
        textTransform: 'none',
        borderRadius: '8px',
      }}
    >
      Execute
    </Button>
    <Button
      variant="outlined"
      color="secondary"
      onClick={handleClear}
      style={{
        padding: '10px 20px',
        fontSize: '16px',
        textTransform: 'none',
        borderRadius: '8px',
      }}
    >
      Clear
    </Button>
  </Box>
      </Grid>
      {response && (
  <Grid item xs={12} style={{ marginTop: '20px' }}>
    {response.map((res, resIndex) => (
      <Alert key={resIndex} severity="info" style={{ marginBottom: '20px' }}>
        <Typography
          variant="h6"
          style={{ fontWeight: 'bold', marginBottom: '10px' }}
        >
          {"CAS: "}
          {fileContent==="" ? options['CAS'][data] : fileContentLst[resIndex]}

        </Typography>
        {toxicityTypes.map((type, index) => (
          <Box
            key={index}
            display="flex"
            flexDirection="row"
            alignItems="center"
            mb={1}
          >
            <Typography
              variant="body1"
              style={{ fontWeight: 'bold', marginRight: '8px' }}
            >
              {type}:
            </Typography>
            {res[type] === 'T' ? (
              <Typography color="error" style={{ fontWeight: 'bold' }}>
                Toxic
              </Typography>
            ) : (
              <Typography style={{ color: 'green', fontWeight: 'bold' }}>
                Non-Toxic
              </Typography>
                )}
              </Box>
            ))}
          </Alert>
        ))}
      </Grid>
    )}
        </Grid>
      </Paper>

          <Grid container spacing={4} className={classes.iconSection} justifyContent="center">
            <Grid item xs={12}>
              <Divider className={classes.divider} />
            </Grid>
            <Grid item xs={12} sm={6} md={2} className={classes.iconWrapper}>
              <img src={LargeDataCollectionImage} alt="Large Data Included" className={classes.icon} />
              <Typography className={classes.iconText}>
                Large Data Included
              </Typography>
              <Typography>
                MultiTox was developed based on chemical structure data, 72 in vitro assay, and 2389 in vivo human organ toxicity data.
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={2} className={classes.iconWrapper}>
              <img src={CuratedDataImage} alt="Data Preprocessing" className={classes.icon} />
              <Typography className={classes.iconText}>
                Data Preprocessing
              </Typography>
              <Typography>
                All collected data were carefully analyzed and utilized to generate molecular descriptors and conduct feature selection.
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={2} className={classes.iconWrapper}>
              <img src={MachineLearningImage} alt="Advanced AI-assisted QSAR Models" className={classes.icon} />
              <Typography className={classes.iconText}>
                Advanced AI-assisted QSAR Models
              </Typography>
              <Typography>
                QSAR models were built using advanced deep learning algorithms, and the process of model development and validation followed OECD guidance to provide reliable and robust predictions.
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={2} className={classes.iconWrapper}>
              <img src={ReliablePredictionsImage} alt="User-Friendly Interface" className={classes.icon} />
              <Typography className={classes.iconText}>
                User-Friendly Interface
              </Typography>
              <Typography>
                This web dashboard is user-friendly for users with and without coding expertise, and users can click on the Tutorial tab to learn how to use this tool.
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Divider className={classes.divider} />
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </>
  );
};

export default Dashboard;
