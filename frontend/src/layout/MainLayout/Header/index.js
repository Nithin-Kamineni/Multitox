/* eslint-disable */
import PropTypes from 'prop-types';

// material-ui
import { useTheme } from '@mui/material/styles';
import { Avatar, Box, ButtonBase, Button } from '@mui/material';
import { makeStyles } from '@mui/styles';

// project imports
import LogoSection from '../LogoSection';
// import SearchSection from './SearchSection';
import ProfileSection from './ProfileSection';
import NotificationSection from './NotificationSection';

import logo from 'assets/logo1.ico'; // Ensure this path is correct

// assets
import { IconMenu2 } from '@tabler/icons-react';

import phhp from 'assets/images/phhp.png';

// ==============================|| MAIN NAVBAR / HEADER ||============================== //

const useStyles = makeStyles((theme) => ({
  topImage: {
    width: '75%',
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
  },
  iconText: {
    marginTop: '10px',
    fontWeight: 'bold',
  },
  divider: {
    height: '100%',
    backgroundColor: theme.palette.primary.main,
  },
}));

const Header = ({ handleLeftDrawerToggle }) => {
  const theme = useTheme();
  const classes = useStyles();

  const handleTutorialClick = () => {
    window.open('https://docs.google.com/document/d/1l7jORCgEjAZnMK09TcVlPruClSKBuKoSX9rC-wTD3Ok/edit?usp=sharing', '_blank'); // Replace with your actual tutorial link
  };

  return (
    <>
      {/* logo & toggler button */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          marginLeft: { lg: theme.spacing(1), xl: theme.spacing(60) },
        }}
      >
        <img src={logo} alt="Data Preprocessing" className={classes.icon} style={{ width: 70, height: 70, marginRight: 5 }} />
        <Box
          sx={{
            width: 228,
            display: 'flex',
            [theme.breakpoints.down('md')]: {
              width: 'auto',
            },
          }}
        >
          <Box component="span" sx={{ display: { xs: 'none', md: 'block' }, flexGrow: 1 }}>
            <LogoSection />
          </Box>
        </Box>
      </Box>

      <Box
        sx={{
          flexGrow: 1,
          marginLeft: { lg: theme.spacing(3), xl: theme.spacing(0) },
          marginRight: { lg: theme.spacing(3), xl: theme.spacing(0) },
        }}
      />

      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          marginRight: { lg: theme.spacing(1), xl: theme.spacing(60) },
        }}
      >
        {/* tutorial button and logo */}
        <Button
          variant="contained"
          color="primary"
          onClick={handleTutorialClick}
          sx={{ marginRight: 5 }}
          size="large"
        >
          Tutorial
        </Button>
        <img src={phhp} alt="Data Preprocessing" className={classes.icon} style={{ width: "auto", height: 50, marginRight: 2 }} />
      </Box>
    </>
  );
};

Header.propTypes = {
  handleLeftDrawerToggle: PropTypes.func
};

export default Header;
