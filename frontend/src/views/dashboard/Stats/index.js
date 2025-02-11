/* eslint-disable */
import { useEffect, useState, useContext } from 'react';

// material-ui
import { Grid } from '@mui/material';

// project imports
import EarningCard from './EarningCard';
import PopularCard from './PopularCard';
import TotalOrderLineChartCard from './TotalOrderLineChartCard';
import TotalIncomeDarkCard from './TotalIncomeDarkCard';
import TotalIncomeLightCard from './TotalIncomeLightCard';
import TotalGrowthBarChart from './TotalGrowthBarChart';
import { gridSpacing } from 'store/constant';
import { Heatmap } from './HeatMap/Heatmap';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import OptionsChoice from 'ui-component/OptionsChoice';
import OutputList from 'ui-component/OutputList';
import { ParametersContext } from 'context';
import { BarChart } from '@mui/x-charts/BarChart';


// ==============================|| DEFAULT DASHBOARD ||============================== //

const Dashboard = () => {

  const { url } = useContext(ParametersContext);

  return (
    <>
      <Grid container spacing={gridSpacing}>
        <Grid container spacing={2} sx={{marginTop:"50px", marginLeft:"20px", marginRight:"20:px"}}>
          stats: {url}
        </Grid>
      </Grid>
    </>
  );
};

export default Dashboard;
