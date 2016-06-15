  /***********************************************************************************************
   ***********************************************************************************************
   **
   ** calculate gsp_XeobarV
   **
   ***********************************************************************************************
   ***********************************************************************************************/

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    sprintf(filename, "gsp_XeobarV.%.4d.aff", Nconf);
    fprintf(stdout, "# [calculate_gsp] writing correlator data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_reader, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[calculate_gsp] Error, aff writer is not initialized\n");
      EXIT(103);
    }
  }
#endif

  /***********************************************
   * calculate Xeobar V
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    X_eo (eo_spinor_field[evecs_num+ievecs], eo_spinor_field[ievecs], -g_mu, gauge_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for X_eo = %e seconds\n", retime-ratime);
  }
  
  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [calculate_gsp] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, g_source_momentum);

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] using source gamma id %d\n", g_source_gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      /* loop on eigenvectors */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        for(kevecs = ievecs; kevecs<evecs_num; kevecs++) {

          ratime = _GET_TIME;
          eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[evecs_num+ievecs], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[evecs_num+kevecs]);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

          ratime = _GET_TIME;
          eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_gsp_momentum_projection = %e\n", retime - ratime);

          for(x0=0; x0<T; x0++) {
            memcpy(gsp[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
          }

        }
      }

      /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
      for(x0=0; x0<T; x0++) {
        for(ievecs = 0; ievecs<evecs_num-1; ievecs++) {
          for(kevecs = ievecs+1; kevecs<evecs_num; kevecs++) {
            gsp[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
            gsp[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
          }
        }
      }

      /***********************************************
       * write gsp to disk
       ***********************************************/
#ifdef HAVE_MPI
      status = gsp_init (&gsp_buffer, 1, 1, T_global, evecs_num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*evecs_num*evecs_num; /* number of items to be sent and received */

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      fprintf(stdout, "# [calculate_gsp] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);
      MPI_Allgather(gsp[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], g_source_gamma_id_list[isource_gamma_id], x0);
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = evecs_num*evecs_num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[calculate_gsp] Error  from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "gsp_XeobarV.%.4d.px%.2dpy%.2dpz%.2d.g%.2d", 
            Nconf, g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[calculate_gsp] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * evecs_num*evecs_num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[calculate_gsp] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */

#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for gamma id %d = %e seconds\n", g_source_gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for momentum (%d, %d, %d) = %e seconds\n",
       g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  /***********************************************
   * close the output files
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
  /* if(aff_status_str != NULL) free(aff_status_str); */
#endif
