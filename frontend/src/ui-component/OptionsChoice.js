
import * as React from 'react';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import { ParametersContext } from 'context';



// ==============================|| LOADER ||============================== //
const OptionsChoice = () => {

    const { Catagory, setCatagory } = React.useContext(ParametersContext);

  const handleChange = (event) => {
    setCatagory(event.target.value);
  };
  return (
        <Box sx={{ minWidth: 120 }}>
        <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">X-axis</InputLabel>
        <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={Catagory}
            label="X-axis"
            onChange={handleChange}
        >
            <MenuItem value={"species"}>Species</MenuItem>
            <MenuItem value={"organs"}>Organs</MenuItem>
        </Select>
        </FormControl>
    </Box>
    );
}

export default OptionsChoice;
